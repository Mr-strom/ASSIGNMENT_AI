# Course & College FAQ Assistant - Evaluation Report

## Summary Scorecard by Prompting Strategy

| Prompting Strategy | Correctness (1-5) | Relevance (1-5) | Consistency (1-5) | Quality (1-5) | **Overall Score (1-5)** | Intent Accuracy | Unknown Declined | Hallucination Rate | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot** | 4.79 | 4.93 | 4.44 | 4.6 | **4.69** | 100.0% | 100.0% | 0.0% | 4.91s |
| **Few-Shot** | 4.88 | 4.93 | 4.25 | 4.5 | **4.64** | 100.0% | 100.0% | 0.0% | 4.96s |
| **Role** | 4.54 | 4.77 | 3.72 | 4.75 | **4.44** | 100.0% | 100.0% | 0.0% | 5.36s |

## Detailed Per-Query Breakdown

| Query ID | Type | Query Text | Zero-Shot Score | Few-Shot Score | Role Score | Intent Matched |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Q01 | in-faq | What is the minimum attendance required to be eligible for final exams? | 5.0 | 5.0 | 5.0 | Yes |
| Q02 | in-faq | What is the penalty for submitting an assignment late? | 3.5 | 5.0 | 2.5 | Yes |
| Q03 | in-faq | What items are allowed in the exam hall during midterms? | 5.0 | 5.0 | 2.5 | Yes |
| Q04 | in-faq | What are the prerequisites and recommended textbooks for CS102 Data Structures? | 5.0 | 5.0 | 5.0 | Yes |
| Q05 | paraphrased | If I am sick and miss 30% of classes, can I still take the end semester tests? | 4.5 | 5.0 | 5.0 | Yes |
| Q06 | paraphrased | Is there any leniency if my homework code is uploaded two hours past midnight? | 5.0 | 5.0 | 5.0 | Yes |
| Q07 | paraphrased | How is our final letter grade determined across homework, midterms, and finals? | 5.0 | 5.0 | 5.0 | Yes |
| Q08 | follow-up | What about for embedded robotics labs? | 5.0 | 5.0 | 5.0 | Yes |
| Q09 | follow-up | What if I return them overdue? | 4.5 | 3.5 | 4.5 | Yes |
| Q10 | unknown | Can I bring my pet python to live with me in the freshman dormitory? | 5.0 | 5.0 | 5.0 | Yes |
| Q11 | unknown | What is the secret recipe for the dining hall's chicken parmesan? | 5.0 | 5.0 | 5.0 | Yes |
| Q12 | unknown | How do I bypass the campus firewall to torrent video games in the dorms? | 5.0 | 5.0 | 5.0 | Yes |