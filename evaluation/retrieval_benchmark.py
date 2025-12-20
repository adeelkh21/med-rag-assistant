"""
Retrieval Evaluation Benchmark Dataset

Manually curated question-answer pairs with relevant chunk IDs.
Used to evaluate BM25, Dense, and Hybrid retrieval quality.

Format:
{
    "questions": [
        {
            "id": int,
            "question": str,
            "category": str,
            "relevant_chunks": [str],  # Expected chunk IDs
            "description": str (optional)
        },
        ...
    ]
}

Categories:
- symptoms
- treatment
- diagnosis
- risk_factors
- prevention
"""

BENCHMARK_DATASET = {
    "questions": [
        # SYMPTOMS
        {
            "id": 1,
            "question": "What are the symptoms of lung cancer?",
            "category": "symptoms",
            "relevant_chunks": ["NCI_CANCER_TREAT_01", "NCI_CANCER_INFO_01", "NCI_NSCLC_TREAT_01"],
            "description": "General lung cancer symptoms"
        },
        {
            "id": 2,
            "question": "What are the signs of uterine sarcoma?",
            "category": "symptoms",
            "relevant_chunks": ["NCI_US_SYM_01"],
            "description": "Uterine sarcoma symptoms"
        },
        {
            "id": 3,
            "question": "What are the symptoms of extragonadal germ cell tumors?",
            "category": "symptoms",
            "relevant_chunks": ["NCI_EGCT_SYM_01"],
            "description": "Extragonadal germ cell tumor symptoms"
        },
        {
            "id": 4,
            "question": "What are the signs of endometrial cancer?",
            "category": "symptoms",
            "relevant_chunks": ["NCI_EC_SYM_01"],
            "description": "Endometrial cancer symptoms"
        },
        {
            "id": 5,
            "question": "What symptoms indicate small cell lung cancer?",
            "category": "symptoms",
            "relevant_chunks": ["NCI_SCLC_DIAG_01"],
            "description": "Small cell lung cancer indicators"
        },
        
        # TREATMENT
        {
            "id": 6,
            "question": "How is uterine sarcoma treated?",
            "category": "treatment",
            "relevant_chunks": ["NCI_US_TREAT_01", "NCI_US_TREAT_02", "NCI_US_TREAT_03", "NCI_US_TREAT_04", "NCI_US_TREAT_05"],
            "description": "Uterine sarcoma treatment options"
        },
        {
            "id": 7,
            "question": "What treatments are available for extragonadal germ cell tumors?",
            "category": "treatment",
            "relevant_chunks": ["NCI_EGCT_TREAT_03", "NCI_EGCT_TREAT_04", "NCI_EGCT_TREAT_05", "NCI_EGCT_TREAT_06"],
            "description": "Extragonadal germ cell tumor treatments"
        },
        {
            "id": 8,
            "question": "How is endometrial cancer treated?",
            "category": "treatment",
            "relevant_chunks": ["NCI_EC_TREAT_01", "NCI_EC_TREAT_02"],
            "description": "Endometrial cancer treatment"
        },
        {
            "id": 9,
            "question": "What is chemotherapy for cancer?",
            "category": "treatment",
            "relevant_chunks": ["NCI_US_TREAT_04", "NCI_EGCT_TREAT_04"],
            "description": "Chemotherapy explanation"
        },
        {
            "id": 10,
            "question": "What is radiation therapy?",
            "category": "treatment",
            "relevant_chunks": ["NCI_US_TREAT_04", "NCI_EGCT_TREAT_03"],
            "description": "Radiation therapy explanation"
        },
        
        # DIAGNOSIS
        {
            "id": 11,
            "question": "How is cancer diagnosed?",
            "category": "diagnosis",
            "relevant_chunks": ["NCI_CANCER_DIAG_01", "NCI_CANCER_DIAG_02", "NCI_CANCER_DIAG_03"],
            "description": "General cancer diagnosis"
        },
        {
            "id": 12,
            "question": "What tests are used to diagnose uterine sarcoma?",
            "category": "diagnosis",
            "relevant_chunks": ["NCI_US_DIAG_01", "NCI_US_STAGE_01"],
            "description": "Uterine sarcoma diagnostic tests"
        },
        {
            "id": 13,
            "question": "How are extragonadal germ cell tumors detected?",
            "category": "diagnosis",
            "relevant_chunks": ["NCI_CANCER_DIAG_04", "NCI_CANCER_DIAG_05"],
            "description": "Extragonadal germ cell tumor detection"
        },
        {
            "id": 14,
            "question": "How is endometrial cancer diagnosed?",
            "category": "diagnosis",
            "relevant_chunks": ["NCI_CANCER_DIAG_06", "NCI_CANCER_DIAG_07"],
            "description": "Endometrial cancer diagnosis"
        },
        {
            "id": 15,
            "question": "What is a biopsy?",
            "category": "diagnosis",
            "relevant_chunks": ["NCI_CANCER_DIAG_03", "NCI_CANCER_DIAG_05"],
            "description": "Biopsy procedure"
        },
        
        # RISK FACTORS
        {
            "id": 16,
            "question": "What are the risk factors for uterine sarcoma?",
            "category": "risk_factors",
            "relevant_chunks": ["NCI_US_RISK_01", "NCI_CANCER_RISK_02"],
            "description": "Uterine sarcoma risk factors"
        },
        {
            "id": 17,
            "question": "What increases the risk of extragonadal germ cell tumors?",
            "category": "risk_factors",
            "relevant_chunks": ["NCI_CANCER_RISK_03"],
            "description": "Extragonadal germ cell tumor risk factors"
        },
        {
            "id": 18,
            "question": "What are the risk factors for endometrial cancer?",
            "category": "risk_factors",
            "relevant_chunks": ["NCI_EC_RISK_01", "NCI_CANCER_RISK_06"],
            "description": "Endometrial cancer risk factors"
        },
        {
            "id": 19,
            "question": "Does obesity increase cancer risk?",
            "category": "risk_factors",
            "relevant_chunks": ["NCI_CANCER_RISK_06"],
            "description": "Obesity and cancer"
        },
        {
            "id": 20,
            "question": "How does estrogen affect cancer risk?",
            "category": "risk_factors",
            "relevant_chunks": ["NCI_CANCER_RISK_04", "NCI_CANCER_RISK_05", "NCI_CANCER_RISK_06"],
            "description": "Estrogen and cancer"
        },
        
        # PREVENTION
        {
            "id": 21,
            "question": "How can endometrial cancer be prevented?",
            "category": "prevention",
            "relevant_chunks": ["NCI_CANCER_PREV_01", "NCI_CANCER_PREV_02", "NCI_CANCER_GEN_01"],
            "description": "Endometrial cancer prevention"
        },
        {
            "id": 22,
            "question": "What are cancer prevention clinical trials?",
            "category": "prevention",
            "relevant_chunks": ["NCI_CANCER_PREV_03"],
            "description": "Clinical trials for prevention"
        },
        
        # STAGING
        {
            "id": 23,
            "question": "What are the stages of uterine sarcoma?",
            "category": "staging",
            "relevant_chunks": ["NCI_US_STAGE_01", "NCI_US_TREAT_02"],
            "description": "Uterine sarcoma staging"
        },
        {
            "id": 24,
            "question": "What are the stages of endometrial cancer?",
            "category": "staging",
            "relevant_chunks": ["NCI_EC_STAGE_01", "NCI_EC_DIAG_01"],
            "description": "Endometrial cancer staging"
        },
        {
            "id": 25,
            "question": "How does cancer spread in the body?",
            "category": "staging",
            "relevant_chunks": ["NCI_US_TREAT_02", "NCI_EGCT_TREAT_02", "NCI_EC_STAGE_01"],
            "description": "Cancer metastasis"
        },
        
        # PROGNOSIS
        {
            "id": 26,
            "question": "What affects prognosis for non-small cell lung cancer?",
            "category": "prognosis",
            "relevant_chunks": ["NCI_NSCLC_TREAT_01"],
            "description": "NSCLC prognosis factors"
        },
        {
            "id": 27,
            "question": "What factors affect uterine sarcoma prognosis?",
            "category": "prognosis",
            "relevant_chunks": ["NCI_US_TREAT_01"],
            "description": "Uterine sarcoma prognosis"
        },
        {
            "id": 28,
            "question": "What affects extragonadal germ cell tumor prognosis?",
            "category": "prognosis",
            "relevant_chunks": ["NCI_EGCT_TREAT_01"],
            "description": "Extragonadal germ cell tumor prognosis"
        },
        {
            "id": 29,
            "question": "What affects endometrial cancer prognosis?",
            "category": "prognosis",
            "relevant_chunks": ["NCI_EC_TREAT_01"],
            "description": "Endometrial cancer prognosis"
        },
        
        # CLINICAL TRIALS
        {
            "id": 30,
            "question": "What are clinical trials for cancer?",
            "category": "clinical_trials",
            "relevant_chunks": ["NCI_CANCER_DIAG_01", "NCI_CANCER_TREAT_03", "NCI_EGCT_INFO_02"],
            "description": "Clinical trials explanation"
        },
        {
            "id": 31,
            "question": "Should I participate in a clinical trial?",
            "category": "clinical_trials",
            "relevant_chunks": ["NCI_CANCER_TREAT_05", "NCI_US_TREAT_05"],
            "description": "Clinical trial participation"
        },
        
        # PROCEDURES
        {
            "id": 32,
            "question": "What is a hysterectomy?",
            "category": "procedures",
            "relevant_chunks": ["NCI_US_STAGE_01", "NCI_EC_TREAT_02"],
            "description": "Hysterectomy procedure"
        },
        {
            "id": 33,
            "question": "What is a CT scan?",
            "category": "procedures",
            "relevant_chunks": ["NCI_CANCER_TREAT_01", "NCI_US_DIAG_01", "NCI_EC_DIAG_01"],
            "description": "CT scan explanation"
        },
        {
            "id": 34,
            "question": "What is a chest x-ray?",
            "category": "procedures",
            "relevant_chunks": ["NCI_CANCER_TREAT_01", "NCI_CANCER_DIAG_04", "NCI_EC_DIAG_01"],
            "description": "Chest x-ray explanation"
        },
        
        # SPECIFIC CONDITIONS
        {
            "id": 35,
            "question": "What is endometrial hyperplasia?",
            "category": "conditions",
            "relevant_chunks": ["NCI_CANCER_RISK_04", "NCI_CANCER_RISK_05"],
            "description": "Endometrial hyperplasia"
        },
        {
            "id": 36,
            "question": "What is Lynch syndrome?",
            "category": "conditions",
            "relevant_chunks": ["NCI_CANCER_RISK_06"],
            "description": "Lynch syndrome and cancer"
        },
        {
            "id": 37,
            "question": "What is Klinefelter syndrome?",
            "category": "conditions",
            "relevant_chunks": ["NCI_CANCER_RISK_03"],
            "description": "Klinefelter syndrome and cancer"
        },
        
        # GENETICS
        {
            "id": 38,
            "question": "Does family history affect cancer risk?",
            "category": "genetics",
            "relevant_chunks": ["NCI_CANCER_GEN_01", "NCI_CANCER_GEN_02"],
            "description": "Family history and cancer"
        },
        {
            "id": 39,
            "question": "What genetic factors increase endometrial cancer risk?",
            "category": "genetics",
            "relevant_chunks": ["NCI_CANCER_RISK_06", "NCI_CANCER_GEN_01"],
            "description": "Genetic risk factors for endometrial cancer"
        },
        
        # FOLLOW-UP
        {
            "id": 40,
            "question": "What are follow-up tests after cancer treatment?",
            "category": "follow_up",
            "relevant_chunks": ["NCI_US_TREAT_05", "NCI_EGCT_INFO_02", "NCI_US_DIAG_02"],
            "description": "Post-treatment follow-up"
        },
        
        # DIVERSE QUERIES
        {
            "id": 41,
            "question": "What is a transvaginal ultrasound?",
            "category": "procedures",
            "relevant_chunks": ["NCI_CANCER_DIAG_03", "NCI_US_DIAG_01", "NCI_CANCER_DIAG_07"],
            "description": "Transvaginal ultrasound"
        },
        {
            "id": 42,
            "question": "What is a pelvic exam?",
            "category": "procedures",
            "relevant_chunks": ["NCI_CANCER_DIAG_02", "NCI_EC_DIAG_01"],
            "description": "Pelvic exam"
        },
        {
            "id": 43,
            "question": "What is tamoxifen and how does it affect cancer risk?",
            "category": "medications",
            "relevant_chunks": ["NCI_CANCER_RISK_06"],
            "description": "Tamoxifen and cancer risk"
        },
        {
            "id": 44,
            "question": "What is hormone therapy for cancer?",
            "category": "treatment",
            "relevant_chunks": ["NCI_US_TREAT_04"],
            "description": "Hormone therapy"
        },
        {
            "id": 45,
            "question": "What is stem cell transplant?",
            "category": "treatment",
            "relevant_chunks": ["NCI_EGCT_TREAT_04", "NCI_CANCER_TREAT_04"],
            "description": "Stem cell transplant"
        },
        
        # COMPLEX MULTI-ASPECT QUERIES
        {
            "id": 46,
            "question": "What procedures diagnose and stage uterine cancer?",
            "category": "diagnosis",
            "relevant_chunks": ["NCI_US_DIAG_01", "NCI_US_STAGE_01", "NCI_CANCER_DIAG_03"],
            "description": "Diagnosis and staging procedures"
        },
        {
            "id": 47,
            "question": "What are tumor markers for germ cell tumors?",
            "category": "diagnosis",
            "relevant_chunks": ["NCI_CANCER_DIAG_04", "NCI_CANCER_DIAG_05", "NCI_EGCT_TREAT_05"],
            "description": "Tumor markers"
        },
        {
            "id": 48,
            "question": "How does pregnancy affect cancer risk?",
            "category": "risk_factors",
            "relevant_chunks": ["NCI_CANCER_PREV_01", "NCI_CANCER_PREV_02", "NCI_CANCER_GEN_01"],
            "description": "Pregnancy and cancer risk"
        },
        {
            "id": 49,
            "question": "What new cancer treatments are being studied?",
            "category": "treatment",
            "relevant_chunks": ["NCI_CANCER_TREAT_02", "NCI_EGCT_TREAT_04", "NCI_US_TREAT_04"],
            "description": "Experimental treatments"
        },
        {
            "id": 50,
            "question": "How are nonseminoma germ cell tumors treated?",
            "category": "treatment",
            "relevant_chunks": ["NCI_EGCT_TREAT_05", "NCI_EGCT_TREAT_06"],
            "description": "Nonseminoma treatment"
        }
    ],
    "metadata": {
        "total_questions": 50,
        "categories": [
            "symptoms", "treatment", "diagnosis", "risk_factors", "prevention",
            "staging", "prognosis", "clinical_trials", "procedures", "conditions",
            "genetics", "follow_up", "medications"
        ],
        "description": "Manually curated benchmark for evaluating medical retrieval quality",
        "purpose": "Compare BM25, Dense, and Hybrid retrieval strategies"
    }
}


def get_benchmark_dataset() -> dict:
    """Get the complete benchmark dataset."""
    return BENCHMARK_DATASET


def get_questions() -> list:
    """Get list of all questions."""
    return BENCHMARK_DATASET["questions"]


def get_question_by_id(question_id: int) -> dict:
    """Get specific question by ID."""
    for q in BENCHMARK_DATASET["questions"]:
        if q["id"] == question_id:
            return q
    return None


def get_questions_by_category(category: str) -> list:
    """Get all questions in a specific category."""
    return [q for q in BENCHMARK_DATASET["questions"] if q["category"] == category]


# Save to JSON file
if __name__ == "__main__":
    import json
    from pathlib import Path
    
    output_path = Path("evaluation/retrieval_benchmark.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(BENCHMARK_DATASET, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Benchmark dataset saved to {output_path}")
    print(f"  Total questions: {len(BENCHMARK_DATASET['questions'])}")
    print(f"  Categories: {len(BENCHMARK_DATASET['metadata']['categories'])}")
