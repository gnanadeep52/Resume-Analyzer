EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {"type": "array", "items": {"type": "string"}},
        "tools": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["skills", "tools"],
}

RESUME_STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_name": {"type": "string"},
        "summary": {"type": "string"},
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "location": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "company", "start_date", "end_date", "bullets"],
            },
        },
    },
    "required": ["candidate_name", "experience"],
}

GAP_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "match_score": {"type": "number"}, 
        "missing": {
            "type": "object",
            "properties": {
                "skills": {"type": "array", "items": {"type": "string"}},
                "tools": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["skills", "tools"],
        },
        "matched": {
            "type": "object",
            "properties": {
                "skills": {"type": "array", "items": {"type": "string"}},
                "tools": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["skills", "tools"],
        },
    },
    "required": ["match_score", "missing", "matched"],
}



ADDENDUM_SCHEMA = {
    "type": "object",
    "properties": {
        "keywords_incorporated": {"type": "array", "items": {"type": "string"}},
        "experience_additions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "company": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "bullets_to_add": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                "required": ["title", "company", "start_date", "end_date", "bullets_to_add"],
            },
        },
        "changes_made": {"type": "string"},
    },
    "required": ["keywords_incorporated", "experience_additions"],
}

ADDENDUM_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "ats_grade": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "timeframe_violations": {"type": "array", "items": {"type": "string"}},
        "keyword_coverage_pct": {"type": "number"},
        "improvement_suggestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["passed", "ats_grade", "issues", "timeframe_violations", "keyword_coverage_pct"],
}
