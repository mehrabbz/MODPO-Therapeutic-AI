# Data Request Form

## Anonymized Research Data for MODPO Therapeutic AI

This repository contains code for "Multi-Objective Alignment of Language Models for Personalized Psychotherapy." Due to the sensitive nature of mental health research data, the datasets used in this study are not publicly released. However, **anonymized data may be shared with qualified researchers** who demonstrate proper credentials for human subjects research.

---

## Available Data

Upon approval, researchers may request access to:

| Dataset | Description | Records |
|---------|-------------|---------|
| **Patient Personas** | Anonymized synthetic personas derived from survey responses (demographics, therapeutic preferences, attitudes toward AI) | 150 personas |
| **Preference Rankings** | LLM-evaluated preference rankings for therapeutic responses across multiple criteria | ~119K rankings |
| **Therapeutic Q&A Pairs** | Questions from EPITOME corpus with generated responses | 2,379 train / 600 test |
| **Model Responses (Test Set)** | Generated responses from all trained models on the 600 test questions | 600 questions × 7+ models |
| **Evaluation Results** | Head-to-head comparison results and win rates | 600 questions × all model pairs |

### Model Responses Detail

The test set model responses include outputs from:
- **Base Model** (Mistral-7B-Instruct-v0.2)
- **GPT-4o** (baseline)
- **SFT Empathy**
- **DPO Empathy**
- **DPO Soup** (parameter merging)
- **Joint-Loss DPO**
- **MODPO Empathy**
- **MODPO Survey** (5 therapeutic criteria + safety)
- **MODPO Survey4** (4 therapeutic criteria + safety)
- **MODPO Maxim** (Gricean maxims + safety)

This allows researchers to replicate evaluation results without running model inference.

**Note:** Raw survey responses and identifiable participant information are **not available** for sharing.

---

## Eligibility Requirements

To request data access, you must demonstrate:

1. **Institutional Affiliation**  
   Active affiliation with a university, research institution, or healthcare organization

2. **Human Subjects Research Training**  
   Current certification in human subjects research ethics, such as:
   - CITI Program (Collaborative Institutional Training Initiative)
   - NIH Protecting Human Research Participants
   - Equivalent institutional training

3. **IRB/Ethics Approval** (if applicable)  
   Documentation that your institution's IRB/ethics board has reviewed your proposed use of the data, OR confirmation that your use qualifies for exemption

4. **Research Purpose**  
   Clear description of how the data will be used for legitimate research purposes

---

## Request Process

### Step 1: Complete the Request Form

Please provide the following information via email:

**Requestor Information:**
- Full Name:
- Title/Position:
- Institution:
- Department:
- Institutional Email:
- ORCID (if available):

**Credentials:**
- [ ] I have completed CITI or equivalent human subjects research training
- [ ] Training completion date:
- [ ] Certificate number (if applicable):

**Research Proposal:**
- Project Title:
- Brief Description (200 words max):
- Specific datasets requested:
- Intended use of data:
- Expected outputs (publications, tools, etc.):

**Data Security:**
- [ ] Data will be stored on institutional/secured systems
- [ ] Data will not be shared with third parties
- [ ] Data will be deleted upon project completion or as specified

**Agreements:**
- [ ] I agree to cite the original paper in any publications using this data
- [ ] I agree to use the data only for the stated research purpose
- [ ] I agree not to attempt to re-identify any participants
- [ ] I agree to notify the authors of any publications resulting from this data

### Step 2: Submit Request

Send your completed request to:

**Email:** [INSERT CONTACT EMAIL]  
**Subject Line:** `MODPO Data Request - [Your Institution]`

Please attach:
- Completed request form (copy the template above)
- Proof of human subjects research training (certificate or screenshot)
- IRB approval letter or exemption confirmation (if applicable)

### Step 3: Review Process

- Requests are typically reviewed within **2-4 weeks**
- You may be contacted for additional information
- Approved requestors will receive a data use agreement (DUA) to sign
- Data will be shared via secure transfer upon DUA execution

---

## Data Use Agreement

Approved requestors must sign a Data Use Agreement that includes:

1. **Purpose Limitation**: Data used only for stated research purpose
2. **No Re-identification**: No attempts to identify individual participants
3. **Security Requirements**: Appropriate data storage and handling
4. **No Redistribution**: Data not shared with unauthorized parties
5. **Publication Requirements**: Citation of original paper; notification of publications
6. **Deletion Timeline**: Data deleted within 1 year of project completion or upon request

---

## Citation

If you use data from this study, please cite:

```bibtex
@article{beikzadeh2025modpo,
  title={Multi-Objective Alignment of Language Models for Personalized Psychotherapy},
  author={Beikzadeh, Mehrab and Malgaroli, Matteo and Gabriel, Saadia},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Contact

For questions about data access or the request process:

**Primary Contact:** Mehrab Beikzadeh  
**Email:** [INSERT EMAIL]  
**Institution:** University of California, Los Angeles

---

## Frequently Asked Questions

**Q: How long does the review process take?**  
A: Typically 2-4 weeks, depending on completeness of application and reviewer availability.

**Q: Can students request data?**  
A: Yes, but requests from students should include faculty advisor information and may require advisor co-signature on the DUA.

**Q: What if my institution doesn't require IRB approval?**  
A: Please provide documentation from your institution explaining why IRB approval is not required (e.g., exemption letter, institutional policy statement).

**Q: Can I request only specific subsets of the data?**  
A: Yes, please specify which datasets you need in your request.

**Q: Is there a cost for data access?**  
A: No, data is provided free of charge for legitimate research purposes.

**Q: Can the data be used for commercial purposes?**  
A: Data is provided for non-commercial research purposes only. Commercial use requires separate negotiation.
