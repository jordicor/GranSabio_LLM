"""
Enhanced Logging System for Gran Sabio LLM Engine
==================================================

Provides visual, colored, structured logging with phase tracking.
IMPORTANT: No emojis in console output (Windows encoding issues).
"""

import logging
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime
from colorama import Fore, Back, Style, init

# Initialize colorama for Windows
init(autoreset=True)

# Phase definitions
class Phase:
    """Phase constants for the generation pipeline"""
    PREFLIGHT = "PREFLIGHT_VALIDATION"
    GENERATION = "CONTENT_GENERATION"
    SMART_EDIT = "SMART_EDIT"
    QA = "QA_EVALUATION"
    CONSENSUS = "CONSENSUS_CALCULATION"
    GRAN_SABIO = "GRAN_SABIO_ESCALATION"
    COMPLETION = "COMPLETION"

# Phase colors
PHASE_COLORS = {
    Phase.PREFLIGHT: Fore.CYAN,
    Phase.GENERATION: Fore.GREEN,
    Phase.SMART_EDIT: Fore.YELLOW,
    Phase.QA: Fore.BLUE,
    Phase.CONSENSUS: Fore.MAGENTA,
    Phase.GRAN_SABIO: Fore.RED,
    Phase.COMPLETION: Fore.GREEN + Style.BRIGHT,
}

# Phase icons (text-based, no emojis for Windows)
PHASE_ICONS = {
    Phase.PREFLIGHT: "[PRE]",
    Phase.GENERATION: "[GEN]",
    Phase.SMART_EDIT: "[EDT]",
    Phase.QA: "[QA ]",
    Phase.CONSENSUS: "[CON]",
    Phase.GRAN_SABIO: "[GS ]",
    Phase.COMPLETION: "[OK ]",
}

# Phase labels for prompts/responses
PHASE_LABELS = {
    Phase.PREFLIGHT: "PREFLIGHT",
    Phase.GENERATION: "GENERATION",
    Phase.SMART_EDIT: "SMART_EDIT",
    Phase.QA: "QA",
    Phase.CONSENSUS: "CONSENSUS",
    Phase.GRAN_SABIO: "GRAN_SABIO",
    Phase.COMPLETION: "COMPLETION",
}


class TimingTracker:
    """Track timing for phases and operations"""

    def __init__(self):
        self._timings: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, key: str):
        """Start timing for a key"""
        self._start_times[key] = time.time()

    def end(self, key: str) -> float:
        """End timing and return elapsed seconds"""
        if key not in self._start_times:
            return 0.0
        elapsed = time.time() - self._start_times[key]
        self._timings[key] = elapsed
        del self._start_times[key]
        return elapsed

    def get(self, key: str) -> Optional[float]:
        """Get timing for a key"""
        return self._timings.get(key)

    def clear(self):
        """Clear all timings"""
        self._timings.clear()
        self._start_times.clear()

    def get_all(self) -> Dict[str, float]:
        """Get all recorded timings"""
        return self._timings.copy()


class PhaseLogger:
    """
    Enhanced logger with phase tracking and visual formatting

    Usage:
        phase_logger = PhaseLogger(session_id="abc123", extra_verbose=True)

        with phase_logger.phase(Phase.GENERATION):
            phase_logger.info("Starting content generation...")
            phase_logger.log_prompt("gpt-4", system_prompt, user_prompt)
            # ... generation logic ...
            phase_logger.log_response("gpt-4", response_content)
    """

    def __init__(
        self,
        session_id: str,
        verbose: bool = False,
        extra_verbose: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        self.session_id = session_id
        self.verbose = verbose
        self.extra_verbose = extra_verbose
        self.logger = logger or logging.getLogger(__name__)
        self.timing_tracker = TimingTracker()
        self._current_phase: Optional[str] = None
        self._phase_stack = []
        self._iteration_context: Optional[int] = None

    def set_iteration(self, iteration: int):
        """Set current iteration number for context"""
        self._iteration_context = iteration

    @contextmanager
    def phase(self, phase_name: str, sub_label: Optional[str] = None):
        """
        Context manager for phase tracking with automatic timing

        Example:
            with phase_logger.phase(Phase.QA, sub_label="Layer 1: Grammar"):
                # QA logic here
                pass
        """
        self._enter_phase(phase_name, sub_label)
        try:
            yield self
        finally:
            self._exit_phase(phase_name)

    def _enter_phase(self, phase_name: str, sub_label: Optional[str] = None):
        """Enter a new phase"""
        self._phase_stack.append(self._current_phase)
        self._current_phase = phase_name

        timing_key = f"phase_{phase_name}_{len(self._phase_stack)}"
        self.timing_tracker.start(timing_key)

        self._print_phase_header(phase_name, sub_label)

    def _exit_phase(self, phase_name: str):
        """Exit current phase"""
        timing_key = f"phase_{phase_name}_{len(self._phase_stack)}"
        elapsed = self.timing_tracker.end(timing_key)

        self._print_phase_footer(phase_name, elapsed)

        self._current_phase = self._phase_stack.pop() if self._phase_stack else None

    def _print_phase_header(self, phase_name: str, sub_label: Optional[str] = None):
        """Print phase header with color and formatting"""
        color = PHASE_COLORS.get(phase_name, Fore.WHITE)
        icon = PHASE_ICONS.get(phase_name, "[???]")

        separator = "=" * 60  # Reduced from 80 to fit better with logging prefix
        timestamp = datetime.now().strftime("%H:%M:%S")

        iteration_str = f" [Iteration {self._iteration_context}]" if self._iteration_context else ""
        sub_str = f" - {sub_label}" if sub_label else ""

        # Print blank line, then separator (two separate calls to avoid formatting issues)
        self.logger.info("")
        self.logger.info(f"{color}{separator}{Style.RESET_ALL}")
        self.logger.info(
            f"{color}{icon} {phase_name}{iteration_str}{sub_str} [{timestamp}]{Style.RESET_ALL}"
        )
        self.logger.info(f"{color}{separator}{Style.RESET_ALL}")

    def _print_phase_footer(self, phase_name: str, elapsed: float):
        """Print phase footer with timing"""
        color = PHASE_COLORS.get(phase_name, Fore.WHITE)
        icon = PHASE_ICONS.get(phase_name, "[???]")

        separator = "-" * 60  # Reduced from 80 to fit better with logging prefix
        elapsed_str = f"{elapsed:.2f}s" if elapsed > 0 else "N/A"

        self.logger.info(f"{color}{separator}{Style.RESET_ALL}")
        self.logger.info(
            f"{color}{icon} {phase_name} COMPLETED (Elapsed: {elapsed_str}){Style.RESET_ALL}"
        )
        self.logger.info(f"{color}{separator}{Style.RESET_ALL}\n")

    def info(self, message: str):
        """Log info message with current phase context"""
        if self._current_phase:
            color = PHASE_COLORS.get(self._current_phase, Fore.WHITE)
            icon = PHASE_ICONS.get(self._current_phase, "[???]")
            self.logger.info(f"{color}{icon}{Style.RESET_ALL} {message}")
        else:
            self.logger.info(message)

    def debug(self, message: str):
        """Log debug message (only if verbose)"""
        if self.verbose:
            self.logger.debug(f"{Fore.WHITE}{Style.DIM}{message}{Style.RESET_ALL}")

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(f"{Fore.YELLOW}[WARN] {message}{Style.RESET_ALL}")

    def error(self, message: str):
        """Log error message"""
        self.logger.error(f"{Fore.RED}{Style.BRIGHT}[ERROR] {message}{Style.RESET_ALL}")

    def log_prompt(
        self,
        model: str,
        system_prompt: Optional[str],
        user_prompt: str,
        **kwargs
    ):
        """
        Log full prompt (only if extra_verbose)

        Args:
            model: Model name
            system_prompt: System prompt
            user_prompt: User prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        """
        if not self.extra_verbose:
            return

        color = PHASE_COLORS.get(self._current_phase, Fore.WHITE)
        separator = "~" * 60  # Reduced from 80 to fit better with logging prefix

        # Get phase label for context
        phase_label = PHASE_LABELS.get(self._current_phase, "UNKNOWN")

        # Print blank line, then separator (separate calls to avoid formatting issues)
        self.logger.info("")
        self.logger.info(f"{color}{separator}{Style.RESET_ALL}")
        self.logger.info(f"{color}[EXTRA_VERBOSE] PROMPT TO {phase_label} ({model}){Style.RESET_ALL}")
        self.logger.info(f"{color}{separator}{Style.RESET_ALL}")

        if system_prompt:
            self.logger.info(f"{Fore.CYAN}[SYSTEM PROMPT]{Style.RESET_ALL}")
            self.logger.info(system_prompt)
            self.logger.info("")

        self.logger.info(f"{Fore.GREEN}[USER PROMPT]{Style.RESET_ALL}")
        self.logger.info(user_prompt)

        if kwargs:
            self.logger.info("")
            self.logger.info(f"{Fore.YELLOW}[PARAMETERS]{Style.RESET_ALL}")
            for key, value in kwargs.items():
                self.logger.info(f"  {key}: {value}")

        self.logger.info(f"{color}{separator}{Style.RESET_ALL}")
        self.logger.info("")

    def log_response(
        self,
        model: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log full response (only if extra_verbose)

        Args:
            model: Model name
            response: Full response content
            metadata: Optional metadata (tokens, timing, etc.)
        """
        if not self.extra_verbose:
            return

        color = PHASE_COLORS.get(self._current_phase, Fore.WHITE)
        separator = "~" * 60  # Reduced from 80 to fit better with logging prefix

        # Get phase label for context
        phase_label = PHASE_LABELS.get(self._current_phase, "UNKNOWN")

        # Print blank line, then separator (separate calls to avoid formatting issues)
        self.logger.info("")
        self.logger.info(f"{color}{separator}{Style.RESET_ALL}")
        self.logger.info(f"{color}[EXTRA_VERBOSE] RESPONSE FROM {phase_label} ({model}){Style.RESET_ALL}")
        self.logger.info(f"{color}{separator}{Style.RESET_ALL}")

        if metadata:
            self.logger.info(f"{Fore.YELLOW}[METADATA]{Style.RESET_ALL}")
            for key, value in metadata.items():
                self.logger.info(f"  {key}: {value}")
            self.logger.info("")

        self.logger.info(f"{Fore.GREEN}[RESPONSE]{Style.RESET_ALL}")
        self.logger.info(response)

        self.logger.info(f"{color}{separator}{Style.RESET_ALL}")
        self.logger.info("")

    def log_content_preview(self, content: str, max_chars: int = 500):
        """Log content preview (only if extra_verbose)"""
        if not self.extra_verbose:
            return

        preview = content[:max_chars]
        if len(content) > max_chars:
            preview += f"\n... ({len(content) - max_chars} more characters)"

        self.logger.info(f"{Fore.CYAN}[CONTENT PREVIEW]{Style.RESET_ALL}")
        self.logger.info(preview)
        self.logger.info("")

    def log_decision(
        self,
        decision: str,
        score: Optional[float] = None,
        reason: Optional[str] = None
    ):
        """
        Log a decision (approved/rejected/etc.)

        Args:
            decision: Decision text (e.g., "APPROVED", "REJECTED")
            score: Optional score
            reason: Optional reason
        """
        if decision.upper() in ["APPROVED", "PROCEED", "PASS"]:
            color = Fore.GREEN + Style.BRIGHT
            icon = "[OK]"
        else:
            color = Fore.RED + Style.BRIGHT
            icon = "[REJECT]"

        score_str = f" (Score: {score:.2f}/10)" if score is not None else ""
        self.logger.info(f"{color}{icon} DECISION: {decision}{score_str}{Style.RESET_ALL}")

        if reason:
            self.logger.info(f"  Reason: {reason}")

    def log_qa_result(
        self,
        model: str,
        score: float,
        is_deal_breaker: bool = False,
        feedback: Optional[str] = None
    ):
        """Log QA evaluation result"""
        if is_deal_breaker:
            icon = "[X]"
            color = Fore.RED + Style.BRIGHT
            status = "DEAL-BREAKER"
        elif score >= 8.0:
            icon = "[+]"
            color = Fore.GREEN
            status = "PASS"
        elif score >= 6.0:
            icon = "[~]"
            color = Fore.YELLOW
            status = "MARGINAL"
        else:
            icon = "[-]"
            color = Fore.RED
            status = "FAIL"

        self.logger.info(
            f"{color}{icon} {model}: {score:.1f}/10 ({status}){Style.RESET_ALL}"
        )

        if feedback and self.verbose:
            # Truncate feedback if too long
            feedback_display = feedback[:200] + "..." if len(feedback) > 200 else feedback
            self.logger.info(f"  Feedback: {feedback_display}")

    def log_timing_summary(self):
        """Log timing summary for all phases (only if extra_verbose)"""
        if not self.extra_verbose:
            return

        timings = self.timing_tracker.get_all()
        if not timings:
            return

        separator = "=" * 60  # Reduced from 80 to fit better with logging prefix
        # Print blank line, then separator (separate calls to avoid formatting issues)
        self.logger.info("")
        self.logger.info(f"{Fore.WHITE}{Style.BRIGHT}{separator}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.WHITE}{Style.BRIGHT}TIMING SUMMARY{Style.RESET_ALL}")
        self.logger.info(f"{Fore.WHITE}{Style.BRIGHT}{separator}{Style.RESET_ALL}")

        total_time = 0.0
        for key, elapsed in sorted(timings.items()):
            # Extract phase name from key
            phase_name = key.replace("phase_", "").rsplit("_", 1)[0]
            color = PHASE_COLORS.get(phase_name, Fore.WHITE)
            self.logger.info(f"{color}{phase_name:30s} {elapsed:8.2f}s{Style.RESET_ALL}")
            total_time += elapsed

        self.logger.info(f"{Fore.WHITE}{Style.BRIGHT}{separator}{Style.RESET_ALL}")
        self.logger.info(f"{Fore.WHITE}{Style.BRIGHT}TOTAL TIME: {total_time:.2f}s{Style.RESET_ALL}")
        self.logger.info(f"{Fore.WHITE}{Style.BRIGHT}{separator}{Style.RESET_ALL}")
        self.logger.info("")


# Convenience functions
def create_phase_logger(
    session_id: str,
    verbose: bool = False,
    extra_verbose: bool = False
) -> PhaseLogger:
    """Create a new PhaseLogger instance"""
    return PhaseLogger(
        session_id=session_id,
        verbose=verbose,
        extra_verbose=extra_verbose
    )
