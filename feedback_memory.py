"""
Feedback Memory System for Gran Sabio LLM Engine
=================================================

Persistent memory system that tracks QA feedback across iterations,
detects patterns using embeddings, and provides intelligent context
for content generation with temporal decay.

Features:
- SQLite persistence for session data
- Embedding-based similarity detection
- Automatic pattern recognition and rule synthesis
- Temporal decay for feedback context
- Cross-session learning for similar requests
"""

import asyncio
import hashlib
import json_utils as json
import logging
import re
import sqlite3
import threading
import time
import zlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path

import aiosqlite
from ai_service import get_ai_service
from config import config

logger = logging.getLogger(__name__)

# Piggyback cleanup configuration
FB_CLEANUP_INTERVAL_SECONDS = 3600  # 1 hour between opportunistic cleanups
FB_CLEANUP_RETENTION_DAYS = 30      # Delete sessions older than 30 days
FB_CLEANUP_ARCHIVE_DAYS = 14        # Archive sessions older than 14 days


# ---------- Configuration ----------

@dataclass
class FeedbackConfig:
    """Configuration for feedback memory system"""
    db_path: str = "feedback_memory.db"
    similarity_threshold: float = 0.86
    norm_threshold: int = 3  # Occurrences before becoming a rule
    max_recent_iterations: int = 30
    retention_days: int = 90
    archive_days: int = 30
    cache_hours: int = 24
    max_evidence_samples: int = 5
    max_rules: int = 15
    embedding_model: str = "text-embedding-3-large"
    analysis_model: str = "gpt-4o-mini"
    analysis_temperature: float = 0.1


# ---------- Utilities ----------

def normalize_text(text: str, max_length: int = 200) -> str:
    """Normalize and truncate text for storage"""
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    return text


def sha1_hash(text: str) -> str:
    """Generate SHA1 hash for text"""
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(x * x for x in vec1) ** 0.5
    norm2 = sum(x * x for x in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# ---------- Database Manager ----------

class FeedbackDatabase:
    """Manages SQLite database for feedback persistence"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._pool = None
        self._last_cleanup_ts: float = 0.0
        self._cleanup_running: bool = False

    async def initialize(self):
        """Initialize database and create schema"""
        self._pool = await aiosqlite.connect(self.db_path)
        await self._pool.execute("PRAGMA journal_mode=WAL")
        await self._pool.execute("PRAGMA synchronous=NORMAL")
        await self._pool.execute("PRAGMA foreign_keys=ON")
        await self._create_schema()
        await self._pool.commit()

    async def _create_schema(self):
        """Create database schema"""
        await self._pool.executescript("""
            -- Session metadata
            CREATE TABLE IF NOT EXISTS session_metadata (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active',  -- active, completed, archived, deleted
                request_hash TEXT,
                total_iterations INTEGER DEFAULT 0,
                final_success BOOLEAN DEFAULT FALSE,
                user_id TEXT,
                metadata_json TEXT
            );

            -- Iteration feedback storage
            CREATE TABLE IF NOT EXISTS iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                iteration_num INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                feedback_text TEXT NOT NULL,
                content_snapshot TEXT,
                summaries_json TEXT,  -- tiered summaries
                issues_json TEXT,      -- extracted issues
                analysis_json TEXT,    -- full analysis result
                FOREIGN KEY (session_id) REFERENCES session_metadata(session_id),
                UNIQUE(session_id, iteration_num)
            );

            -- Feedback categories/patterns
            CREATE TABLE IF NOT EXISTS feedback_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                canonical_label TEXT NOT NULL,
                category_type TEXT,
                occurrences INTEGER DEFAULT 1,
                severity TEXT,  -- high, medium, low
                evidence_json TEXT,  -- sample quotes
                actions_json TEXT,   -- corrective actions
                embedding_json TEXT, -- vector embedding
                first_seen INTEGER,
                last_seen INTEGER,
                FOREIGN KEY (session_id) REFERENCES session_metadata(session_id),
                UNIQUE(session_id, concept_id)
            );

            -- Normative rules derived from patterns
            CREATE TABLE IF NOT EXISTS normative_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                rule_text TEXT NOT NULL,
                source_patterns_json TEXT,
                creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                active BOOLEAN DEFAULT TRUE
            );

            -- Indices for performance
            CREATE INDEX IF NOT EXISTS idx_session_status
                ON session_metadata(status, last_activity);
            CREATE INDEX IF NOT EXISTS idx_session_hash
                ON session_metadata(request_hash);
            CREATE INDEX IF NOT EXISTS idx_iterations_session
                ON iterations(session_id, iteration_num);
            CREATE INDEX IF NOT EXISTS idx_categories_session
                ON feedback_categories(session_id, occurrences DESC);
            CREATE INDEX IF NOT EXISTS idx_categories_concept
                ON feedback_categories(session_id, concept_id);

            CREATE INDEX IF NOT EXISTS idx_normative_rules_lookup
                ON normative_rules(session_id, active, creation_date DESC);

            CREATE INDEX IF NOT EXISTS idx_session_hash_success
                ON session_metadata(request_hash, final_success);
        """)

    async def close(self):
        """Close database connection"""
        if self._pool:
            await self._pool.close()

    # Session Management

    async def create_session(self, session_id: str, request_hash: str, metadata: Dict = None):
        """Create new session entry"""
        await self._pool.execute("""
            INSERT INTO session_metadata
            (session_id, request_hash, metadata_json, status)
            VALUES (?, ?, ?, 'active')
        """, (session_id, request_hash, json.dumps(metadata or {})))
        await self._pool.commit()
        asyncio.create_task(self._maybe_piggyback_cleanup())

    async def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session metadata"""
        async with self._pool.execute("""
            SELECT session_id, created_at, last_activity, status,
                   request_hash, total_iterations, final_success, metadata_json
            FROM session_metadata WHERE session_id = ?
        """, (session_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    'session_id': row[0],
                    'created_at': row[1],
                    'last_activity': row[2],
                    'status': row[3],
                    'request_hash': row[4],
                    'total_iterations': row[5],
                    'final_success': row[6],
                    'metadata': json.loads(row[7]) if row[7] else {}
                }
        return None

    async def update_session_status(self, session_id: str, status: str, success: bool = None):
        """Update session status"""
        if success is not None:
            await self._pool.execute("""
                UPDATE session_metadata
                SET status = ?, final_success = ?, last_activity = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (status, success, session_id))
        else:
            await self._pool.execute("""
                UPDATE session_metadata
                SET status = ?, last_activity = CURRENT_TIMESTAMP
                WHERE session_id = ?
            """, (status, session_id))
        await self._pool.commit()
        asyncio.create_task(self._maybe_piggyback_cleanup())

    async def find_similar_sessions(self, request_hash: str, limit: int = 5) -> List[str]:
        """Find sessions with similar request hash"""
        async with self._pool.execute("""
            SELECT session_id
            FROM session_metadata
            WHERE request_hash = ?
            AND final_success = TRUE
            AND status != 'deleted'
            ORDER BY created_at DESC
            LIMIT ?
        """, (request_hash, limit)) as cursor:
            return [row[0] for row in await cursor.fetchall()]

    # Iteration Management

    async def add_iteration(self, session_id: str, iteration_num: int,
                           feedback_text: str, content_snapshot: str,
                           summaries: Dict, issues: List, analysis: Dict):
        """Add iteration feedback to database"""
        await self._pool.execute("""
            INSERT INTO iterations
            (session_id, iteration_num, feedback_text, content_snapshot,
             summaries_json, issues_json, analysis_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, iteration_num, feedback_text, content_snapshot,
              json.dumps(summaries), json.dumps(issues), json.dumps(analysis)))

        # Update session iteration count
        await self._pool.execute("""
            UPDATE session_metadata
            SET total_iterations = ?, last_activity = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (iteration_num + 1, session_id))

        await self._pool.commit()
        asyncio.create_task(self._maybe_piggyback_cleanup())

    async def get_recent_iterations(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent iterations for a session"""
        async with self._pool.execute("""
            SELECT iteration_num, timestamp, feedback_text, content_snapshot,
                   summaries_json, issues_json, analysis_json
            FROM iterations
            WHERE session_id = ?
            ORDER BY iteration_num DESC
            LIMIT ?
        """, (session_id, limit)) as cursor:
            iterations = []
            for row in await cursor.fetchall():
                iterations.append({
                    'iteration_num': row[0],
                    'timestamp': row[1],
                    'feedback_text': row[2],
                    'content_snapshot': row[3],
                    'summaries': json.loads(row[4]) if row[4] else {},
                    'issues': json.loads(row[5]) if row[5] else [],
                    'analysis': json.loads(row[6]) if row[6] else {}
                })
            return iterations

    # Category Management

    async def upsert_category(self, session_id: str, concept_id: str,
                             canonical_label: str, category_type: str,
                             severity: str, evidence: List[str],
                             actions: List[str], embedding: List[float],
                             total_iterations: Optional[int] = None):
        """Insert or update feedback category.

        Args:
            total_iterations: Pre-fetched iteration count. When provided,
                skips the per-call SELECT on session_metadata (batch optimization).
        """
        # Check if exists
        async with self._pool.execute("""
            SELECT occurrences, evidence_json, actions_json
            FROM feedback_categories
            WHERE session_id = ? AND concept_id = ?
        """, (session_id, concept_id)) as cursor:
            existing = await cursor.fetchone()

        # Use pre-fetched value when available, otherwise fetch individually
        current_iteration = total_iterations if total_iterations is not None else await self._get_current_iteration(session_id)

        if existing:
            # Update existing
            occurrences = existing[0] + 1
            prev_evidence = json.loads(existing[1]) if existing[1] else []
            prev_actions = json.loads(existing[2]) if existing[2] else []

            # Merge and limit samples
            new_evidence = (prev_evidence + evidence)[:5]
            new_actions = (prev_actions + actions)[:5]

            await self._pool.execute("""
                UPDATE feedback_categories
                SET occurrences = ?,
                    evidence_json = ?,
                    actions_json = ?,
                    embedding_json = ?,
                    last_seen = ?,
                    severity = ?
                WHERE session_id = ? AND concept_id = ?
            """, (occurrences, json.dumps(new_evidence), json.dumps(new_actions),
                  json.dumps(embedding), current_iteration, severity,
                  session_id, concept_id))
        else:
            # Insert new
            await self._pool.execute("""
                INSERT INTO feedback_categories
                (session_id, concept_id, canonical_label, category_type,
                 occurrences, severity, evidence_json, actions_json,
                 embedding_json, first_seen, last_seen)
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
            """, (session_id, concept_id, canonical_label, category_type,
                  severity, json.dumps(evidence), json.dumps(actions),
                  json.dumps(embedding), current_iteration, current_iteration))

    async def get_categories(self, session_id: str, min_occurrences: int = 1) -> List[Dict]:
        """Get feedback categories for session"""
        async with self._pool.execute("""
            SELECT concept_id, canonical_label, category_type, occurrences,
                   severity, evidence_json, actions_json, embedding_json,
                   first_seen, last_seen
            FROM feedback_categories
            WHERE session_id = ? AND occurrences >= ?
            ORDER BY occurrences DESC, last_seen DESC
        """, (session_id, min_occurrences)) as cursor:
            categories = []
            for row in await cursor.fetchall():
                categories.append({
                    'concept_id': row[0],
                    'canonical_label': row[1],
                    'category_type': row[2],
                    'occurrences': row[3],
                    'severity': row[4],
                    'evidence': json.loads(row[5]) if row[5] else [],
                    'actions': json.loads(row[6]) if row[6] else [],
                    'embedding': json.loads(row[7]) if row[7] else None,
                    'first_seen': row[8],
                    'last_seen': row[9]
                })
            return categories

    async def _get_current_iteration(self, session_id: str) -> int:
        """Get current iteration number for session"""
        async with self._pool.execute("""
            SELECT total_iterations FROM session_metadata WHERE session_id = ?
        """, (session_id,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    # Rules Management

    async def save_normative_rules(self, session_id: str, rules: List[str],
                                  source_patterns: List[str]):
        """Save normative rules derived from patterns"""
        await self._pool.execute("""
            INSERT INTO normative_rules
            (session_id, rule_text, source_patterns_json)
            VALUES (?, ?, ?)
        """, (session_id, '\n'.join(rules), json.dumps(source_patterns)))
        await self._pool.commit()
        asyncio.create_task(self._maybe_piggyback_cleanup())

    async def get_active_rules(self, session_id: str) -> List[str]:
        """Get active normative rules for session"""
        async with self._pool.execute("""
            SELECT rule_text
            FROM normative_rules
            WHERE session_id = ? AND active = TRUE
            ORDER BY creation_date DESC
            LIMIT 1
        """, (session_id,)) as cursor:
            row = await cursor.fetchone()
            if row and row[0]:
                return row[0].split('\n')
            return []

    # Cleanup

    async def _maybe_piggyback_cleanup(self):
        """Run cleanup opportunistically if enough time has passed."""
        now = time.monotonic()
        if self._cleanup_running or (now - self._last_cleanup_ts) < FB_CLEANUP_INTERVAL_SECONDS:
            return
        self._cleanup_running = True
        self._last_cleanup_ts = now
        try:
            await self.cleanup_old_sessions(
                retention_days=FB_CLEANUP_RETENTION_DAYS,
                archive_days=FB_CLEANUP_ARCHIVE_DAYS,
            )
            logger.info("Piggyback cleanup: feedback memory cleanup completed")
        except Exception:
            logger.exception("Feedback piggyback cleanup failed")
        finally:
            self._cleanup_running = False

    async def cleanup_old_sessions(self, retention_days: int, archive_days: int):
        """Clean up old sessions based on retention policy"""
        cutoff_archive = datetime.now() - timedelta(days=archive_days)
        cutoff_delete = datetime.now() - timedelta(days=retention_days)

        # Archive old sessions
        await self._pool.execute("""
            UPDATE session_metadata
            SET status = 'archived'
            WHERE status = 'completed'
            AND last_activity < ?
        """, (cutoff_archive,))

        # Delete very old sessions
        await self._pool.execute("""
            DELETE FROM iterations
            WHERE session_id IN (
                SELECT session_id FROM session_metadata
                WHERE last_activity < ?
            )
        """, (cutoff_delete,))

        await self._pool.execute("""
            DELETE FROM feedback_categories
            WHERE session_id IN (
                SELECT session_id FROM session_metadata
                WHERE last_activity < ?
            )
        """, (cutoff_delete,))

        await self._pool.execute("""
            DELETE FROM normative_rules
            WHERE session_id IN (
                SELECT session_id FROM session_metadata
                WHERE last_activity < ?
            )
        """, (cutoff_delete,))

        await self._pool.execute("""
            DELETE FROM session_metadata
            WHERE last_activity < ?
        """, (cutoff_delete,))

        await self._pool.commit()


# ---------- Feedback Processor ----------

class FeedbackProcessor:
    """Process and analyze QA feedback"""

    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.ai_service = get_ai_service()

    async def extract_feedback_analysis(self, feedback_text: str) -> Dict[str, Any]:
        """Extract structured analysis from feedback text using AI"""

        prompt = f"""Analyze this QA consensus feedback and extract structured information.

QA FEEDBACK:
{feedback_text}

Provide a JSON response with:

1. "tiered_summaries": Create summaries at different detail levels
   - "lines_5": Array of up to 5 bullet points (most detailed)
   - "lines_3": Array of up to 3 bullet points
   - "lines_2": Array of up to 2 bullet points
   - "one_liner": Single sentence capturing the core issue

2. "issues": Array of atomic issues, each with:
   - "canonical_label": Short canonical name (e.g., "missing_dates", "excessive_adjectives")
   - "abstract": One-sentence description
   - "type": One of ["facts", "completeness", "structure", "style", "accuracy", "format", "logic"]
   - "severity": "high", "medium", or "low"
   - "evidence_quote": Short quote from feedback (max 200 chars)
   - "action": Specific corrective instruction (imperative mood)

3. "next_iteration_hint": 1-3 sentences instructing how to fix the main issues

Return ONLY valid JSON, no additional text."""

        try:
            response = await self.ai_service.generate_content(
                prompt=prompt,
                model=self.config.analysis_model,
                temperature=self.config.analysis_temperature,
                json_output=True
            )

            # Parse and validate response
            if isinstance(response, str):
                analysis = json.loads(response)
            else:
                analysis = response

            # Ensure required fields
            if 'tiered_summaries' not in analysis:
                analysis['tiered_summaries'] = self._create_fallback_summaries(feedback_text)
            if 'issues' not in analysis:
                analysis['issues'] = []
            if 'next_iteration_hint' not in analysis:
                analysis['next_iteration_hint'] = "Address the feedback points above."

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze feedback: {e}")
            return {
                'tiered_summaries': self._create_fallback_summaries(feedback_text),
                'issues': [],
                'next_iteration_hint': "Address the feedback points above."
            }

    def _create_fallback_summaries(self, text: str) -> Dict[str, Any]:
        """Create fallback summaries if AI analysis fails"""
        sentences = text.split('.')[:5]
        return {
            'lines_5': sentences[:5],
            'lines_3': sentences[:3],
            'lines_2': sentences[:2],
            'one_liner': sentences[0] if sentences else "Feedback provided"
        }

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for text list"""
        try:
            # Use the AI service to get embeddings
            response = await self.ai_service._make_request(
                'POST',
                'https://api.openai.com/v1/embeddings',
                json={
                    'model': self.config.embedding_model,
                    'input': texts
                }
            )

            if response and 'data' in response:
                return [item['embedding'] for item in response['data']]

        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")

        return [[] for _ in texts]  # Return empty embeddings on failure

    async def synthesize_normative_rules(self, categories: List[Dict]) -> List[str]:
        """Synthesize normative rules from repeated patterns"""

        if not categories:
            return []

        # Build prompt with top categories
        patterns_text = "\n".join([
            f"- {cat['canonical_label']} (occurred {cat['occurrences']} times): {cat['actions'][0] if cat['actions'] else ''}"
            for cat in categories[:10]
        ])

        prompt = f"""Based on these recurring QA issues, create imperative rules for the content generator.

RECURRING ISSUES:
{patterns_text}

Create up to {self.config.max_rules} concise DO/DON'T rules that will prevent these issues.
Focus on the most frequent and severe issues.

Return a JSON object with a single field "rules" containing an array of rule strings.
Each rule should be imperative and actionable (e.g., "ALWAYS include publication dates", "AVOID excessive superlatives").

Return ONLY valid JSON."""

        try:
            response = await self.ai_service.generate_content(
                prompt=prompt,
                model=self.config.analysis_model,
                temperature=0.0,
                json_output=True
            )

            if isinstance(response, str):
                result = json.loads(response)
            else:
                result = response

            return result.get('rules', [])[:self.config.max_rules]

        except Exception as e:
            logger.error(f"Failed to synthesize rules: {e}")
            # Fallback: create simple rules from labels
            return [
                f"Address issue: {cat['canonical_label']}"
                for cat in categories[:5]
            ]


# ---------- Main Feedback Memory Manager ----------

class FeedbackMemoryManager:
    """Main manager for feedback memory system"""

    def __init__(self, config: Optional[FeedbackConfig] = None):
        self.config = config or FeedbackConfig()
        self.db = FeedbackDatabase(self.config.db_path)
        self.processor = FeedbackProcessor(self.config)
        self.memory_cache = {}  # In-memory cache for active sessions
        self._initialized = False

    async def initialize(self):
        """Initialize the feedback memory system"""
        if not self._initialized:
            await self.db.initialize()
            self._initialized = True
            logger.info(f"Feedback memory initialized with database: {self.config.db_path}")

    async def close(self):
        """Close database connections"""
        await self.db.close()

    def _hash_request(self, request: Any) -> str:
        """Generate hash for request to identify similar sessions"""
        # Extract key fields that define uniqueness
        key_parts = [
            str(request.prompt),
            str(request.qa_layers) if hasattr(request, 'qa_layers') else '',
            str(request.generator_model) if hasattr(request, 'generator_model') else '',
            str(request.content_type) if hasattr(request, 'content_type') else ''
        ]
        return sha1_hash('|'.join(key_parts))

    async def initialize_session(self, session_id: str, request: Any) -> Dict[str, Any]:
        """Initialize feedback memory for a new session"""

        # Ensure initialized
        await self.initialize()

        request_hash = self._hash_request(request)

        # Create session in database
        await self.db.create_session(session_id, request_hash, {
            'generator_model': getattr(request, 'generator_model', ''),
            'content_type': getattr(request, 'content_type', '')
        })

        # Find similar successful sessions
        similar_sessions = await self.db.find_similar_sessions(request_hash)

        # Extract common patterns from similar sessions
        initial_rules = []
        if similar_sessions:
            common_patterns = await self._extract_common_patterns(similar_sessions)
            if common_patterns:
                initial_rules = await self.processor.synthesize_normative_rules(common_patterns)

        # Initialize cache entry
        self.memory_cache[session_id] = {
            'rules': initial_rules,
            'categories': {},
            'recent_summaries': [],
            'iteration_count': 0
        }

        logger.info(f"Initialized feedback memory for session {session_id} with {len(initial_rules)} initial rules")

        return {
            'initial_rules': initial_rules,
            'similar_sessions': len(similar_sessions)
        }

    async def _extract_common_patterns(self, session_ids: List[str]) -> List[Dict]:
        """Extract common patterns from multiple sessions"""

        all_categories = []
        pattern_counts = {}

        for session_id in session_ids:
            categories = await self.db.get_categories(session_id, min_occurrences=2)
            for cat in categories:
                label = cat['canonical_label']
                if label not in pattern_counts:
                    pattern_counts[label] = {
                        'count': 0,
                        'total_occurrences': 0,
                        'category': cat
                    }
                pattern_counts[label]['count'] += 1
                pattern_counts[label]['total_occurrences'] += cat['occurrences']

        # Return patterns that appear in multiple sessions
        common = []
        for label, data in pattern_counts.items():
            if data['count'] >= 2:  # Appears in at least 2 sessions
                cat = data['category']
                cat['cross_session_occurrences'] = data['total_occurrences']
                common.append(cat)

        # Sort by cross-session occurrences
        common.sort(key=lambda x: x['cross_session_occurrences'], reverse=True)

        return common[:20]  # Top 20 patterns

    async def add_iteration_feedback(self, session_id: str, feedback_text: str,
                                    content_snapshot: str, iteration_num: int) -> str:
        """Process and store feedback for an iteration"""

        # Extract structured analysis from feedback
        analysis = await self.processor.extract_feedback_analysis(feedback_text)

        summaries = analysis['tiered_summaries']
        issues = analysis['issues']

        # Store in database
        await self.db.add_iteration(
            session_id, iteration_num, feedback_text,
            normalize_text(content_snapshot, 500),
            summaries, issues, analysis
        )

        # Process issues and update categories
        if issues:
            await self._process_issues(session_id, issues, iteration_num)

        # Update cache
        if session_id in self.memory_cache:
            self.memory_cache[session_id]['recent_summaries'] = summaries
            self.memory_cache[session_id]['iteration_count'] = iteration_num + 1

        # Build and return the iteration prompt
        return await self.build_iteration_prompt(session_id)

    async def _process_issues(self, session_id: str, issues: List[Dict], iteration_num: int):
        """Process issues, detect similarities, and update categories"""

        # Get embeddings for all issues
        issue_texts = [
            f"{issue.get('type', 'general')}::{issue.get('canonical_label', '')}"
            for issue in issues
        ]
        embeddings = await self.processor.get_embeddings(issue_texts)

        # Get existing categories for similarity comparison
        existing_categories = await self.db.get_categories(session_id)

        # Cache total_iterations once before the loop to avoid N redundant SELECTs
        total_iterations = await self.db._get_current_iteration(session_id)

        # Process each issue
        for idx, issue in enumerate(issues):
            canonical_label = issue.get('canonical_label', 'unspecified')
            category_type = issue.get('type', 'general')
            severity = issue.get('severity', 'medium')
            evidence = [issue.get('evidence_quote', '')]
            actions = [issue.get('action', '')]
            embedding = embeddings[idx] if idx < len(embeddings) else []

            # Find most similar existing category
            concept_id = sha1_hash(f"{category_type}::{canonical_label}")

            if embedding and existing_categories:
                best_match = None
                best_similarity = 0

                for cat in existing_categories:
                    if cat['embedding']:
                        similarity = cosine_similarity(embedding, cat['embedding'])
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = cat

                # Merge with existing if similarity exceeds threshold
                if best_match and best_similarity >= self.config.similarity_threshold:
                    concept_id = best_match['concept_id']
                    canonical_label = best_match['canonical_label']  # Keep original label

            # Update or insert category (no per-issue commit, batched below)
            await self.db.upsert_category(
                session_id, concept_id, canonical_label,
                category_type, severity, evidence, actions, embedding,
                total_iterations=total_iterations
            )

        # Single commit after all issues have been processed
        await self.db._pool.commit()
        asyncio.create_task(self.db._maybe_piggyback_cleanup())

    async def build_iteration_prompt(self, session_id: str) -> str:
        """Build the iteration feedback prompt with temporal decay"""

        # Get recent iterations
        iterations = await self.db.get_recent_iterations(
            session_id,
            limit=self.config.max_recent_iterations
        )

        # Get categories for pattern rules
        categories = await self.db.get_categories(
            session_id,
            min_occurrences=self.config.norm_threshold
        )

        # Get or synthesize normative rules
        if categories:
            rules = await self.processor.synthesize_normative_rules(categories)
            await self.db.save_normative_rules(
                session_id, rules,
                [cat['canonical_label'] for cat in categories[:10]]
            )
        else:
            rules = await self.db.get_active_rules(session_id)

        # Build prompt sections
        sections = []

        # 1. Core rules section
        if rules:
            sections.append("=== CORE RULES (from repeated issues) ===")
            for rule in rules[:self.config.max_rules]:
                sections.append(f"- {rule}")
            sections.append("")

        # 2. Recurring issues section
        if categories:
            active_categories = [
                cat for cat in categories
                if cat['last_seen'] >= len(iterations) - 5  # Active in last 5 iterations
            ]

            if active_categories:
                sections.append("=== RECURRING ISSUES ===")
                for cat in active_categories[:10]:
                    severity_marker = "🔴" if cat['severity'] == 'high' else "🟡" if cat['severity'] == 'medium' else "⚪"
                    sections.append(
                        f"{severity_marker} {cat['canonical_label']} "
                        f"(failed {cat['occurrences']}x, last in iteration {cat['last_seen']})"
                    )
                sections.append("")

        # 3. Recent feedback with decay
        if iterations:
            sections.append("=== RECENT FEEDBACK (with temporal decay) ===")

            for idx, iteration in enumerate(iterations):
                summaries = iteration['summaries']

                if idx == 0:  # Most recent - full feedback
                    sections.append(f"\nCURRENT FEEDBACK (Iteration {iteration['iteration_num']}) - Full:")
                    sections.append(iteration['feedback_text'])

                elif idx == 1:  # Previous - 5 lines
                    if 'lines_5' in summaries:
                        sections.append(f"\nPREVIOUS (Iteration {iteration['iteration_num']}) - 5 key points:")
                        for line in summaries['lines_5']:
                            sections.append(f"  • {line}")

                elif idx == 2:  # 2 iterations ago - 3 lines
                    if 'lines_3' in summaries:
                        sections.append(f"\nIteration {iteration['iteration_num']} - 3 points:")
                        for line in summaries['lines_3']:
                            sections.append(f"  • {line}")

                elif idx == 3:  # 3 iterations ago - 2 lines
                    if 'lines_2' in summaries:
                        sections.append(f"\nIteration {iteration['iteration_num']} - 2 points:")
                        for line in summaries['lines_2']:
                            sections.append(f"  • {line}")

                else:  # Older - one liner
                    if 'one_liner' in summaries:
                        sections.append(f"Iter {iteration['iteration_num']}: {summaries['one_liner']}")

            sections.append("")

        # 4. Iteration statistics
        if self.memory_cache.get(session_id):
            cache = self.memory_cache[session_id]
            iteration_count = cache.get('iteration_count', 0)

            if categories:
                most_common = categories[0]
                sections.append("=== ITERATION STATISTICS ===")
                sections.append(f"- Current iteration: {iteration_count}")
                sections.append(f"- Total unique issues: {len(categories)}")
                sections.append(f"- Most frequent issue: {most_common['canonical_label']} ({most_common['occurrences']}x)")

                # Improvement detection
                if len(iterations) >= 2:
                    recent_issues = len(iterations[0].get('issues', []))
                    previous_issues = len(iterations[1].get('issues', []))
                    if recent_issues < previous_issues:
                        sections.append(f"- Improvement detected: {previous_issues} → {recent_issues} issues")

                sections.append("")

        # 5. Next iteration focus (from latest analysis)
        if iterations and iterations[0].get('analysis'):
            hint = iterations[0]['analysis'].get('next_iteration_hint')
            if hint:
                sections.append("=== MANDATORY NEXT ITERATION FOCUS ===")
                sections.append(hint)
                sections.append("")

        # 6. Motivational/warning messages
        if categories:
            high_frequency = [cat for cat in categories if cat['occurrences'] >= 5]
            if high_frequency:
                sections.append("⚠️ CRITICAL: The following issues have failed 5+ times:")
                for cat in high_frequency[:3]:
                    sections.append(f"  - {cat['canonical_label']} (failed {cat['occurrences']} times!)")
                sections.append("These MUST be addressed in this iteration.")

        return "\n".join(sections)

    async def complete_session(self, session_id: str, success: bool):
        """Mark session as completed"""

        status = 'completed' if success else 'failed'
        await self.db.update_session_status(session_id, status, success)

        # Clear from cache
        if session_id in self.memory_cache:
            del self.memory_cache[session_id]

        logger.info(f"Session {session_id} completed with status: {status}")

    async def cleanup_old_data(self):
        """Run cleanup for old sessions"""
        await self.db.cleanup_old_sessions(
            self.config.retention_days,
            self.config.archive_days
        )
        logger.info("Completed cleanup of old feedback data")


# ---------- Singleton Instance ----------

_feedback_manager = None

def get_feedback_manager() -> FeedbackMemoryManager:
    """Get or create the singleton feedback manager instance"""
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = FeedbackMemoryManager()
    return _feedback_manager


# ---------- Integration Helpers ----------

async def initialize_feedback_system():
    """Initialize the feedback system on startup"""
    manager = get_feedback_manager()
    await manager.initialize()
    logger.info("Feedback memory system initialized")
    return manager


async def shutdown_feedback_system():
    """Shutdown the feedback system cleanly"""
    manager = get_feedback_manager()
    if manager:
        await manager.close()
    logger.info("Feedback memory system shutdown complete")