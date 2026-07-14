"""
tools.py — Three callable tools for the BVRIT FAQ Chatbot.

Tools:
  1. fee_calculator        — tuition, hostel, scholarship discounts
  2. date_checker          — days remaining / passed for admission/exam dates
  3. percentage_calculator — scholarship %, placement %, cutoff %

Each tool:
  - Has a JSON schema definition for LLM function-calling
  - Validates all inputs and raises ValueError on bad data
  - Returns a plain-text string the LLM can embed in its response
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Fee data (sourced from knowledge base — AY 2025-26)
# ---------------------------------------------------------------------------

TUITION_FEES: dict[str, float] = {
    "cse":        120000,
    "cse-aiml":   135000,
    "cse-ds":     135000,
    "ece":        110000,
    "eee":        100000,
    "mech":       100000,
    "mechanical": 100000,
    "it":         110000,
}

HOSTEL_FEE_3_SEATER: float  = 60000   # room rent per year (3-seater)
HOSTEL_FEE_2_SEATER: float  = 75000   # room rent per year (2-seater)
MESS_VEG_PER_YEAR: float    = 48000   # vegetarian mess per year
MESS_NONVEG_PER_YEAR: float = 54000   # non-vegetarian mess per year
TRANSPORT_FEE_PER_YEAR: float = 45000
LAB_FEE_PER_YEAR: float       = 15000
LIBRARY_FEE_PER_YEAR: float   = 8000
EXAM_FEE_PER_YEAR: float      = 10000  # 2 semesters × 5000
ACTIVITY_FEE_PER_YEAR: float  = 3000
ONE_TIME_FEES: float          = 15000  # caution deposit + admission processing

SCHOLARSHIP_SCHEMES: dict[str, float] = {
    "founders":            100.0,   # EAMCET rank top 1,000 — full waiver
    "academic_excellence":  50.0,   # EAMCET rank 1,001–5,000
    "merit_reward":         25.0,   # EAMCET rank 5,001–15,000
    "sports":               25.0,   # national/state level sports
    "ews":                  50.0,   # family income < Rs.8 lakh/year
    "sc_st":               100.0,   # full tuition reimbursement (state govt)
    "bc":                  100.0,   # full tuition reimbursement (state govt)
    "sibling":              10.0,   # second sibling enrolled at BVRIT
}

# ---------------------------------------------------------------------------
# Input sanitisation helper
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = re.compile(
    r"(ignore|forget|disregard|override|bypass|reveal|system prompt|jailbreak)",
    re.IGNORECASE,
)

def _check_injection(value: str) -> None:
    """Raise ValueError if input looks like a prompt injection attempt."""
    if _INJECTION_PATTERNS.search(str(value)):
        raise ValueError(
            "Invalid input: the value contains disallowed instructions. "
            "Please provide a valid input."
        )

def _validate_branch(branch: str) -> str:
    cleaned = branch.strip().lower().replace(" ", "-").replace("_", "-")
    # Aliases
    aliases = {
        "aiml": "cse-aiml", "ai&ml": "cse-aiml", "ai ml": "cse-aiml",
        "ds": "cse-ds", "data science": "cse-ds",
        "mechanical": "mech",
        "computer science": "cse",
        "information technology": "it",
        "electronics": "ece",
        "electrical": "eee",
    }
    cleaned = aliases.get(cleaned, cleaned)
    if cleaned not in TUITION_FEES:
        valid = "CSE, CSE-AIML, CSE-DS, ECE, EEE, Mech, IT"
        raise ValueError(
            f"Unknown branch '{branch}'. Valid options: {valid}."
        )
    return cleaned

def _validate_scholarship(scheme: str) -> str:
    cleaned = scheme.strip().lower().replace(" ", "_").replace("-", "_")
    # Aliases
    aliases = {
        "founder":    "founders",
        "merit":      "merit_reward",
        "excellence": "academic_excellence",
        "sc":         "sc_st",
        "st":         "sc_st",
    }
    cleaned = aliases.get(cleaned, cleaned)
    if cleaned not in SCHOLARSHIP_SCHEMES:
        valid = "founders, academic_excellence, merit_reward, sports, ews, sc_st, bc, sibling"
        raise ValueError(
            f"Unknown scholarship scheme '{scheme}'. Valid options: {valid}."
        )
    return cleaned

def _validate_years(years: Any) -> int:
    try:
        y = int(years)
    except (TypeError, ValueError):
        raise ValueError(f"'years' must be a whole number, got '{years}'.")
    if not (1 <= y <= 4):
        raise ValueError(f"'years' must be between 1 and 4, got {y}.")
    return y

def _validate_percentage(value: Any, label: str = "percentage") -> float:
    try:
        p = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{label}' must be a number, got '{value}'.")
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"'{label}' must be between 0 and 100, got {p}.")
    return p

def _validate_date_str(date_str: str) -> date:
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse date '{date_str}'. Use formats: YYYY-MM-DD or DD-MM-YYYY."
    )

# ---------------------------------------------------------------------------
# Tool 1 — fee_calculator
# ---------------------------------------------------------------------------

def fee_calculator(
    branch: str,
    years: int = 1,
    include_hostel: bool = False,
    hostel_type: str = "3-seater",
    mess_type: str = "vegetarian",
    include_transport: bool = False,
    include_lab: bool = True,
    scholarship_scheme: str | None = None,
) -> str:
    """
    Calculate total fee for a given branch, duration, and optional extras.

    Args:
        branch:             Department name (e.g. 'CSE', 'ECE', 'EEE').
        years:              Number of years (1–4). Default 1.
        include_hostel:     Add hostel room + mess fee. Default False.
        hostel_type:        '3-seater' or '2-seater'. Default '3-seater'.
        mess_type:          'vegetarian' or 'non-vegetarian'. Default 'vegetarian'.
        include_transport:  Add transport fee (Rs.45,000/year). Default False.
        include_lab:        Add lab fee (Rs.15,000/year). Default True.
        scholarship_scheme: Optional scheme: founders, academic_excellence,
                            merit_reward, sports, ews, sc_st, bc, sibling.

    Returns:
        Formatted fee breakdown string.
    """
    _check_injection(branch)
    if scholarship_scheme:
        _check_injection(scholarship_scheme)

    branch_key = _validate_branch(branch)
    years_val  = _validate_years(years)
    tuition_pa = TUITION_FEES[branch_key]

    # Scholarship on tuition only
    discount_pct   = 0.0
    discount_label = ""
    if scholarship_scheme:
        scheme_key     = _validate_scholarship(scholarship_scheme)
        discount_pct   = SCHOLARSHIP_SCHEMES[scheme_key]
        discount_label = f" ({scheme_key.replace('_',' ').title()} scheme)"

    tuition_after  = tuition_pa * (1 - discount_pct / 100)
    total_tuition  = tuition_after * years_val

    # Hostel room
    if include_hostel:
        ht = hostel_type.strip().lower()
        if "2" in ht:
            room_pa = HOSTEL_FEE_2_SEATER
            room_label = "2-seater"
        else:
            room_pa = HOSTEL_FEE_3_SEATER
            room_label = "3-seater"
        mess_pa    = MESS_NONVEG_PER_YEAR if "non" in mess_type.lower() else MESS_VEG_PER_YEAR
        mess_label = "Non-Vegetarian" if "non" in mess_type.lower() else "Vegetarian"
        hostel_total = (room_pa + mess_pa) * years_val
    else:
        hostel_total = 0.0
        room_label   = ""
        mess_label   = ""

    transport_total = TRANSPORT_FEE_PER_YEAR * years_val if include_transport else 0.0
    lab_total       = LAB_FEE_PER_YEAR * years_val if include_lab else 0.0

    # Fixed annual fees (library, exam, activity) + one-time
    fixed_annual = (LIBRARY_FEE_PER_YEAR + EXAM_FEE_PER_YEAR + ACTIVITY_FEE_PER_YEAR) * years_val
    one_time     = ONE_TIME_FEES if years_val >= 1 else 0.0

    grand_total = total_tuition + hostel_total + transport_total + lab_total + fixed_annual + one_time

    lines = [
        f"💰 **Fee Breakdown — {branch.upper()}** ({years_val} year{'s' if years_val > 1 else ''})",
        "",
        f"• Tuition fee (per year):       ₹{tuition_pa:,.0f}",
    ]
    if discount_pct > 0:
        lines.append(
            f"• Scholarship{discount_label}: -{discount_pct:.0f}%  →  ₹{tuition_after:,.0f}/year"
        )
    lines.append(f"• Tuition total ({years_val} yr):        ₹{total_tuition:,.0f}")

    if include_hostel:
        lines.append(f"• Hostel room ({room_label}, {years_val} yr): ₹{(HOSTEL_FEE_3_SEATER if room_label == '3-seater' else HOSTEL_FEE_2_SEATER) * years_val:,.0f}")
        lines.append(f"• Mess ({mess_label}, {years_val} yr):     ₹{(MESS_VEG_PER_YEAR if 'Veg' in mess_label else MESS_NONVEG_PER_YEAR) * years_val:,.0f}")
    if include_transport:
        lines.append(f"• Transport ({years_val} yr):              ₹{transport_total:,.0f}")
    if include_lab:
        lines.append(f"• Lab fee ({years_val} yr):                ₹{lab_total:,.0f}")
    lines.append(f"• Library + Exam + Activity:        ₹{fixed_annual:,.0f}")
    lines.append(f"• One-time fees (caution+processing):₹{one_time:,.0f}")

    lines += [
        "",
        f"**Grand Total: ₹{grand_total:,.0f}**",
        "",
        "_Fees per AY 2025-26. Confirm with Accounts Office before payment._",
        "_Transport (₹45,000/yr) excluded unless requested._",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2 — date_checker
# ---------------------------------------------------------------------------

KNOWN_DATES: dict[str, str] = {
    "ts_eamcet_exam":                "2025-05-15",
    "eamcet_results":                "2025-06-05",
    "counselling_round1_start":      "2025-07-01",
    "counselling_round1_end":        "2025-07-15",
    "counselling_round2_start":      "2025-07-20",
    "counselling_round2_end":        "2025-07-31",
    "round2_reporting_deadline":     "2025-08-05",
    "management_quota_deadline":     "2025-08-15",
    "orientation":                   "2025-08-18",
    "classes_commence":              "2025-08-19",
    "late_admission_deadline":       "2025-08-31",
    "midsem_exam1_start":            "2025-10-15",
    "midsem_exam1_end":              "2025-10-20",
    "endsem_odd_start":              "2025-12-01",
    "endsem_odd_end":                "2025-12-15",
    "even_semester_start":           "2026-01-05",
    "midsem_exam2_start":            "2026-03-15",
    "midsem_exam2_end":              "2026-03-20",
    "endsem_even_start":             "2026-05-01",
    "endsem_even_end":               "2026-05-15",
}

def date_checker(
    event_name: str | None = None,
    target_date: str | None = None,
) -> str:
    """
    Check how many days remain until (or since) an admission/exam date.

    Args:
        event_name:  Known event key (e.g. 'eamcet_exam', 'last_date_admission').
                     Pass None if supplying target_date directly.
        target_date: An explicit date string (YYYY-MM-DD or DD-MM-YYYY).
                     Used when event_name is None.

    Returns:
        Days-remaining or days-passed message.
    """
    if event_name:
        _check_injection(event_name)
    if target_date:
        _check_injection(target_date)

    # Resolve date
    if event_name:
        key = event_name.strip().lower().replace(" ", "_")
        if key not in KNOWN_DATES:
            valid = ", ".join(KNOWN_DATES.keys())
            raise ValueError(
                f"Unknown event '{event_name}'. Known events: {valid}."
            )
        resolved_date = _validate_date_str(KNOWN_DATES[key])
        label = event_name.replace("_", " ").title()
    elif target_date:
        resolved_date = _validate_date_str(target_date)
        label = target_date
    else:
        raise ValueError("Provide either 'event_name' or 'target_date'.")

    today     = date.today()
    delta     = (resolved_date - today).days

    if delta > 0:
        status = f"🟢 **{delta} day{'s' if delta != 1 else ''} remaining**"
        msg    = f"The deadline has NOT passed yet."
    elif delta == 0:
        status = "🟡 **Today is the deadline!**"
        msg    = "The event is today."
    else:
        status = f"🔴 **Deadline passed {abs(delta)} day{'s' if abs(delta) != 1 else ''} ago**"
        msg    = "The deadline has already passed."

    return (
        f"📅 **Date Check — {label}**\n\n"
        f"• Event date: {resolved_date.strftime('%d %B %Y')}\n"
        f"• Today:      {today.strftime('%d %B %Y')}\n"
        f"• Status:     {status}\n"
        f"• {msg}"
    )


# ---------------------------------------------------------------------------
# Tool 3 — percentage_calculator
# ---------------------------------------------------------------------------

def percentage_calculator(
    calculation_type: str,
    numerator: float | None = None,
    denominator: float | None = None,
    base_amount: float | None = None,
    percentage: float | None = None,
) -> str:
    """
    Calculate scholarship amounts, placement percentages, or cutoff percentages.

    Modes (calculation_type):
      'scholarship_amount'   — percentage% of base_amount
      'placement_percentage' — (numerator / denominator) * 100
      'cutoff_percentage'    — (numerator / denominator) * 100  (marks/total)
      'discount_amount'      — discount rupees from base_amount at percentage%

    Args:
        calculation_type: One of the modes above.
        numerator:   Placed students / marks obtained (for % calculations).
        denominator: Total students / total marks.
        base_amount: Fee or salary base (for amount calculations).
        percentage:  Scholarship or discount percentage (for amount calculations).

    Returns:
        Formatted result string.
    """
    _check_injection(calculation_type)

    calc = calculation_type.strip().lower().replace(" ", "_")
    valid_types = {"scholarship_amount", "placement_percentage", "cutoff_percentage", "discount_amount"}

    if calc not in valid_types:
        raise ValueError(
            f"Unknown calculation_type '{calculation_type}'. "
            f"Valid options: {', '.join(sorted(valid_types))}."
        )

    if calc in ("scholarship_amount", "discount_amount"):
        if base_amount is None or percentage is None:
            raise ValueError("'base_amount' and 'percentage' are required for this calculation.")
        base  = float(base_amount)
        pct   = _validate_percentage(percentage, "percentage")
        if base < 0:
            raise ValueError(f"'base_amount' must be non-negative, got {base}.")
        amount = base * pct / 100
        remaining = base - amount

        if calc == "scholarship_amount":
            return (
                f"🎓 **Scholarship Calculation**\n\n"
                f"• Base amount:          ₹{base:,.0f}\n"
                f"• Scholarship rate:     {pct:.1f}%\n"
                f"• Scholarship amount:   ₹{amount:,.0f}\n"
                f"• **Amount payable:     ₹{remaining:,.0f}**"
            )
        else:
            return (
                f"💸 **Discount Calculation**\n\n"
                f"• Original amount:  ₹{base:,.0f}\n"
                f"• Discount rate:    {pct:.1f}%\n"
                f"• Discount amount:  ₹{amount:,.0f}\n"
                f"• **Payable amount: ₹{remaining:,.0f}**"
            )

    elif calc in ("placement_percentage", "cutoff_percentage"):
        if numerator is None or denominator is None:
            raise ValueError("'numerator' and 'denominator' are required for this calculation.")
        num = float(numerator)
        den = float(denominator)
        if den <= 0:
            raise ValueError(f"'denominator' must be greater than 0, got {den}.")
        if num < 0:
            raise ValueError(f"'numerator' must be non-negative, got {num}.")
        if num > den:
            raise ValueError(
                f"'numerator' ({num}) cannot exceed 'denominator' ({den}). "
                "Check your input values."
            )
        result_pct = (num / den) * 100

        if calc == "placement_percentage":
            return (
                f"📊 **Placement Percentage**\n\n"
                f"• Students placed:  {num:.0f}\n"
                f"• Total students:   {den:.0f}\n"
                f"• **Placement %:    {result_pct:.2f}%**"
            )
        else:
            return (
                f"📝 **Cutoff Percentage**\n\n"
                f"• Marks obtained:  {num:.1f}\n"
                f"• Total marks:     {den:.1f}\n"
                f"• **Percentage:    {result_pct:.2f}%**"
            )

    # Should never reach here
    raise ValueError("Unhandled calculation type.")


# ---------------------------------------------------------------------------
# Tool registry — JSON schemas for LLM function-calling
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "fee_calculator",
            "description": (
                "Calculate tuition fees, hostel fees, transport fees, and "
                "scholarship discounts for BVRIT Hyderabad (AY 2025-26). Use this when the "
                "user asks 'how much will it cost', 'total fee for CSE', "
                "'fee with hostel', or 'fee after scholarship'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "branch": {
                        "type": "string",
                        "description": "Branch name: CSE, CSE-AIML, CSE-DS, ECE, EEE, Mech/Mechanical, IT.",
                    },
                    "years": {
                        "type": "integer",
                        "description": "Number of years 1–4. Default 1.",
                        "minimum": 1,
                        "maximum": 4,
                    },
                    "include_hostel": {
                        "type": "boolean",
                        "description": "Include hostel room + mess fee. Default false.",
                    },
                    "hostel_type": {
                        "type": "string",
                        "description": "Room type: '3-seater' (Rs.60,000/yr) or '2-seater' (Rs.75,000/yr). Default 3-seater.",
                    },
                    "mess_type": {
                        "type": "string",
                        "description": "Mess preference: 'vegetarian' (Rs.48,000/yr) or 'non-vegetarian' (Rs.54,000/yr). Default vegetarian.",
                    },
                    "include_transport": {
                        "type": "boolean",
                        "description": "Include transport fee Rs.45,000/year. Default false.",
                    },
                    "include_lab": {
                        "type": "boolean",
                        "description": "Include lab fee Rs.15,000/year. Default true.",
                    },
                    "scholarship_scheme": {
                        "type": "string",
                        "description": "Scholarship scheme: founders, academic_excellence, merit_reward, sports, ews, sc_st, bc, sibling. Omit if none.",
                    },
                },
                "required": ["branch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "date_checker",
            "description": (
                "Check how many days remain until (or since) an admission or exam deadline. "
                "Use this for questions like 'when is EAMCET', 'how many days left for admission', "
                "'has the counselling deadline passed', or any date-related query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_name": {
                        "type": "string",
                        "description": (
                            "Known event key: ts_eamcet_exam, eamcet_results, "
                            "counselling_round1_start, counselling_round1_end, "
                            "counselling_round2_start, counselling_round2_end, "
                            "round2_reporting_deadline, management_quota_deadline, "
                            "orientation, classes_commence, late_admission_deadline, "
                            "midsem_exam1_start, midsem_exam1_end, "
                            "endsem_odd_start, endsem_odd_end, "
                            "even_semester_start, midsem_exam2_start, midsem_exam2_end, "
                            "endsem_even_start, endsem_even_end."
                        ),
                    },
                    "target_date": {
                        "type": "string",
                        "description": "Explicit date string YYYY-MM-DD or DD-MM-YYYY. Use if event_name is not applicable.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "percentage_calculator",
            "description": (
                "Calculate scholarship amounts, placement percentages, or cutoff percentages. "
                "Use for questions like 'what is 25% scholarship on CSE fee', "
                "'what percentage of students were placed', or 'what is my cutoff percentage'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "calculation_type": {
                        "type": "string",
                        "description": (
                            "Type of calculation: 'scholarship_amount', 'placement_percentage', "
                            "'cutoff_percentage', or 'discount_amount'."
                        ),
                    },
                    "numerator": {
                        "type": "number",
                        "description": "Marks obtained or students placed (for percentage calculations).",
                    },
                    "denominator": {
                        "type": "number",
                        "description": "Total marks or total students (for percentage calculations).",
                    },
                    "base_amount": {
                        "type": "number",
                        "description": "Base fee or salary in rupees (for amount calculations).",
                    },
                    "percentage": {
                        "type": "number",
                        "description": "Scholarship or discount percentage 0–100 (for amount calculations).",
                    },
                },
                "required": ["calculation_type"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher — called by generate.py
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS: dict[str, Any] = {
    "fee_calculator":        fee_calculator,
    "date_checker":          date_checker,
    "percentage_calculator": percentage_calculator,
}

def dispatch_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name and return its string result (or error message)."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'."
    try:
        return fn(**arguments)
    except ValueError as exc:
        return f"⚠️ Input error: {exc}"
    except Exception as exc:
        return f"⚠️ Tool error: {exc}"
