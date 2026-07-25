# Final Project Report

Source: https://northeastern.instructure.com/courses/253590/assignments/3203466

## Assignment Metadata

- **Due:** Sunday by 8:55pm
- **Points:** 100
- **Submitting:** a file upload
- **File Types:** pdf, ipynb, and zip

## Assignment Details

1.  ### (100 Points) Project Report - Due July 26

    Written report formatted as a scientific paper, maximum 8 pages (11 pt font, single-spaced), excluding references and appendices.

    **Format:** Submit as PDF (no Word documents)

    **Required Sections:**

    1.  **Authors:** Names of all group members

    2.  **Introduction:**

        - Background of the problem and why it's important
        - Non-technical summary of your solution

    3.  **Methods:**

        - Describe your implementation approach
        - Include technical details of AI models, APIs, and algorithms used

    4.  **Dataset/Inputs:**

        - Describe your data sources or user inputs
        - How data is processed and used in your application
        - Cite all data sources

    5.  **Results:**

        - Show your working application with screenshots or figures
        - Demonstrate key functionality
        - Keep description concise

    6.  **Discussion:**

        - Compare with existing solutions or state-of-the-art approaches
        - What worked well and what challenges did you face?
        - What did you learn from this project?
        - How could the application be improved in the future?

    7.  **AI Prompts Used (REQUIRED if used):**

        - Include ALL prompts used with Claude or other AI tools
        - Document prompts for: coding, debugging, explanations, research
        - This section is mandatory for grading

    8.  **References:**

        - Only include references explicitly used in the text
        - Cite all data sources and APIs
        - Use consistent formatting (e.g., \[1\] Author. Title. Conference/Journal, Year.)

    9.  **Appendix:**

        - Link to public GitHub repository with code
        - Additional plots or technical details
        - Installation and setup instructions

    10. **Statement of Contributions:**

        - For groups of 2-3: describe each member's contributions
        - Not required for solo projects
        - Does not count toward page limit

    **Code Repository:**

    - Must include link to public GitHub repository
    - If code cannot be made public, submit as zip file on Canvas

    **Writing Standards:**

    - Formal academic writing (no slang or jargon)
    - No spelling or grammar mistakes
    - All claims must be supported by citations, results, or common knowledge

    **Note:** A single submission per group is sufficient.

## Rubric

| Criterion | Points | Description | Perfect | Good | Fair | Bad | No Marks |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| Introduction | 5 | (Why would people want to study this dataset and what is the primary task. Find out what the "target" variable means and identify 2-3 challenges in analyzing the dataset.) 2-3 paragraphs | 5 pts: Clear motivation, stated the problem, possibly tied this to broader content | 3.75 pts: Some details about the problem, little context, missing one or more components, or have everything but not in paragraph form | 2.5 pts: Superficial, too short, or no details | 1.25 pts: Not in paragraph form, just bullets or lists | 0 pts |
| Proposed Method | 5 | Present an 1-paragraph description of your method and why you believe it is better that the other things you have tried | 5 pts: Satisfied everything in the criteria | 3.75 pts: Satisfies but not in paragraph form | 2.5 pts: One or more missing | 1.25 pts: Two or more missing | 0 pts |
| Related Work | 11 | (Find (5-6) examples of people who have worked on similar dataset from the literature. Note: Literature == Published paper in a conference (not stack overflow). Briefly describe in 1-2 sentences the kinds of features, algorithms, or other methods they applied. Also explain why you believe your method is better. Provide a numbered reference id that will appear later in the references section. 3-5 paragraphs | 11 pts: Detailed analysis of 5-6 related works, in paragraph form and detailed explanation. | 8.25 pts: Less than 5-6, but still in paragraph form | 5.5 pts: Detailed but not in paragraph form | 2.75 pts: Superficial analysis or not in paragraph form | 0 pts |
| Related implementations | 11 | (Find (2-3) examples of what people in Kaggle have done on this particular dataset [2]. Reference the URL of their kernel, post, etc. Describe in 1-2 sentences what they have done and why you think your method is better.) 2-3 paragraphs | 11 pts: Detailed analysis of 2-3 related works, in paragraph form and detailed explanation. Includes references. | 8.25 pts: Less than 2-3, but still in paragraph form | 5.5 pts: Detailed but not in paragraph form or superficial | 2.75 pts: Superficial analysis or not in paragraph form | 0 pts |
| Data Analysis | 8 | (Data Analysis: Describe the data analysis you have completed, include 1-2 plots of the most useful features or learnings you have obtained from the dataset. Do not include the code, but do include formulas to anything you have calculated such as different feature combinations, feature selection, or analysis methods.) 5-6 paragraphs | 8 pts: Details of related analysis, described features in detail, used features transformations. A logical solution. | 6 pts: Superficial analysis. Some of the transformations may not make sense, but got the main points. | 4 pts: Analysis had clearly missing components, was superficial. | 2 pts: Not in paragraph form, maybe a list | 0 pts |
| Proposed Method | 8 | (Describe the ML algorithms you used. Focus on the formulas, any feature extractions, parameter tuning, etc. Explain how the algorithm works. E.g., if you used a decision, don't say "I used a decision tree", explain briefly how a decision tree works and why it was ideally suited for the dataset you chose.) 3-5 paragraphs | 8 pts: Explains model well, formulas, clear explanation. In paragraph form of at least 5 detailed paragraphs. | 6 pts: Superficial analysis. Missing formulas, got some of the minor details wrong. Missed some steps. | 4 pts: Incorrect assumptions, missed obvious issues. | 2 pts: Not in paragraph form, maybe a list. Clear reasoning errors. | 0 pts |
| Analysis | 15 | Provide some insights as to why you think that the proposed algorithms and features are good for this dataset. Explain whether you believe these are general properties that might be helpful for similar datasets--what makes them similar and why. What about this dataset made your solution successful. Could we use this for other datasets, if so, what types and why?) 3-5 paragraphs | 15 pts: Good explanation, correct insights. In paragraph forms with 5 detailed paragraphs, Doesn't just recount the results. | 11.25 pts: Good detail but a little light on content, less than 5 detailed paragraphs. | 7.5 pts: Did not provide much insight into the solution, just described what was done. Presented an opinion rather than fact. OR Less than 3 paragraphs, missing details, | 3.75 pts: Not in paragraph form or multiple missing issues | 0 pts |
| Experimental Setup | 8 | (Did you use all the data, cross-validation, training / test split, etc? Give enough details on how you setup the experiment so that your colleague can read this section and write their own algorithm to reproduce the same setup. Provide a link to the cells in the notebooks that contain the experimental setup.) 3-4 paragraphs | 8 pts: Detailed explanation of setup, explained in words. Logical. In paragraph form with 4 good paragraphs. | 6 pts: Less than 3 paragraphs, but otherwise coherent and logical. | 4 pts: High-level details missing or superficial but still in paragraph form. | 2 pts: Not in paragraph form, maybe a list. | 0 pts |
| Results | 11 | (Write a table containing the results of your experiments, which were calculated in the notebooks. Provide some interpretation of these results. Do you think you could have done better? If so, why did you not pursue those ideas? Add any plots you think appropriate here.) 5-6 paragraphs | 11 pts: Detailed results, includes table, figures, and 5-6 detailed paragraphs. Conclusions make sense and are substantial rather than superficial. | 8.25 pts: Less than 5 but more than 3 paragraphs. Generally logical and coherent, but maybe missing some minor details. | 5.5 pts: Shows results without context or interpretation. | 2.75 pts: No analysis, not in paragraph form. | 0 pts |
| Conclusion | 4 | Summarize your findings. If someone wanted to use your solution, which would you recommend? What could you do if you had more data, etc? What should a company seeking to run this at a high scale choose if they were to use your method.) 3 paragraphs | 4 pts: Summarizes work, gives correct guidance, recommendations. Three paragraphs. | 3 pts: Less than 3 paragraphs, but mostly complete. | 2 pts: Superficial analysis | 1 pts: Not in paragraph form or terse | 0 pts |
| References | 4 | References | 4 pts: At least 6 references: Kaggle and papers. Correctly formatted and used within the text. | 3 pts: Less than 5 references, reasonable formatting. | 2 pts: Less than 3 references or bad formatting. | 1 pts: Weird formatting, does not cite papers. | 0 pts |
| Completion | 10 | Review the overall style | 10 pts: High quality, good content, student tried hard | 7.5 pts: Good effort, but were missing some minor points. | 5 pts: Superficial analysis, slapped together in a hurry. | 2.5 pts: Made a few bad choices, did not understand what they were doing. | 0 pts |
