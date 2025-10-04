import os
import json
from groq import Groq
from app.config import settings
from typing import Dict, Any

class SummarizeService:
    """
    A service to handle summary generation using the Groq AI model.
    """

    SYSTEM_PROMPT = """You are a world-class research analyst AI. Your task is to analyze a user's query and generate a response as Markdown text only.

Do not include any JSON, text outside of Markdown, or any additional explanations. Your entire output must be valid Markdown, ready to display in a document or dashboard.

The value of the "summary" key must be a string containing a comprehensive, long-form summary in Markdown format. This summary must be structured with the following sections:
- **Introduction:** Briefly introduce the topic and the scope of the analysis.
- **Key Points:** Use bullet points to detail the core findings, focusing on aspects like scientific progress, knowledge gaps, and areas of consensus or disagreement.
- **Conclusion and Actionable Insights:** Summarize the findings and provide clear, actionable insights for mission planners, researchers, or strategists.


Note:
**Make sure that you use <br> tag instead of \n everywhere.
---
**Example 1:**

User Query: "Impact of AI on climate change modeling"

AI Response:

  "# Summary: Impact of AI on Climate Change Modeling<br><br>**Introduction**<br>Artificial Intelligence (AI) is revolutionizing climate change modeling by enhancing predictive accuracy, processing vast datasets, and identifying complex patterns. This analysis explores the key scientific progress, existing knowledge gaps, and actionable insights for leveraging AI in climate research.<br><br>**Key Points**<br>* **Scientific Progress:** Machine learning models have significantly improved the resolution of climate simulations and the prediction of extreme weather events.<br>* **Knowledge Gaps:** A significant gap exists in understanding the 'black box' nature of some complex AI models, making it difficult to interpret their reasoning.<br>* **Areas of Disagreement:** Experts disagree on the extent to which AI can replace physics-based models, with many advocating for a hybrid approach.<br><br>**Conclusion and Actionable Insights**<br>AI presents a powerful tool for advancing climate change understanding. Actionable steps include prioritizing funding for open-source climate datasets, developing standards for AI model transparency, and fostering collaboration between research institutions."


---
**Example 2:**

User Query: "Latest advancements in battery technology for electric vehicles"

AI Response:
  "# Summary: Advancements in EV Battery Technology<br><br>**Introduction**<br>The rapid evolution of battery technology is a critical driver for the widespread adoption of electric vehicles (EVs). This summary covers recent breakthroughs, the consensus on future directions, and provides actionable insights for the automotive industry.<br><br>**Key Points**<br>* **Scientific Progress:** Solid-state batteries are emerging as a leading next-generation technology, promising higher energy density, improved safety, and faster charging times.<br>* **Areas of Consensus:** There is a strong consensus on reducing dependency on cobalt, a costly and ethically challenging material. Research is heavily focused on chemistries like lithium-iron-phosphate (LFP).<br>* **Knowledge Gaps:** Scaling the manufacturing of solid-state batteries to an industrial level remains a major hurdle.<br><br>**Conclusion and Actionable Insights**<br>The trajectory of battery technology is set towards safer, cheaper, and more energy-dense solutions. Actionable insights include securing supply chains for next-generation materials, investing in R&D for scalable manufacturing, and developing robust battery recycling programs."

"""

    def __init__(self):
        """
        Initializes the Groq client, sets the model, and warms up the service.
        """
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
        
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = "llama-3.3-70b-versatile"
        self.fast_model = "llama-3.1-8b-instant" # A faster model for simple classification
        print("SummarizeService initialized.")
        # self.warmup()

    def warmup(self):
        """
        Performs a simple test call to the Groq API to ensure the model is ready.
        """
        print("Warming up the summarization model...")
        try:
            self.client.chat.completions.create(
                messages=[{"role": "user", "content": "Test call"}],
                model=self.model,
                max_tokens=10,
                temperature=0.1
            )
            print("Summarization model is warm and ready.")
            return True
        except Exception as e:
            print(f"Error during model warmup: {e}")
            return False

    def _get_visualization_system_prompt(self) -> str:
        """
        Returns the system prompt for the AI that generates Chart.js JSON data.
        """
        # This prompt is a detailed instruction set for the AI.
        # It specifies the exact JSON structure, preferred chart types, and rules,
        # ensuring the output is consistently machine-readable for Chart.js.
        return """
        You are an expert data visualization assistant. Your task is to analyze a user's query and a text summary to generate a JSON object suitable for Chart.js.

        Instructions:
        1. Read the user's query and the provided summary carefully.
        2. Identify the key data points, labels, and numerical values that can be visualized.
        3. Choose the most appropriate chart type from: 'bar', 'line', 'pie', 'doughnut', 'radar', or 'polarArea'. A 'bar' chart is often a good default choice for comparisons.
        4. Construct a JSON object that strictly follows the Chart.js configuration format.
        5. Your entire response MUST be a single, valid JSON object and nothing else. Do not include explanations, comments, or markdown formatting like ```json.

        The JSON object MUST have these top-level keys: "type", "data", "options".

        - The "data" object must contain "labels" (an array of strings) and "datasets" (an array of objects).
        - Each object in "datasets" must contain a "label" (a string) and "data" (an array of numbers).
        - Generate appropriate RGBA colors for `backgroundColor` and `borderColor` for better visuals.
        - Add a descriptive title to the chart under `options.plugins.title`.

        Example of a valid response:
        {
        "type": "bar",
        "data": {
            "labels": ["Q1", "Q2", "Q3", "Q4"],
            "datasets": [{
            "label": "Sales 2025 (in millions)",
            "data": [120, 190, 150, 210],
            "backgroundColor": "rgba(54, 162, 235, 0.5)",
            "borderColor": "rgba(54, 162, 235, 1)",
            "borderWidth": 1
            }]
        },
        "options": {
            "responsive": true,
            "plugins": {
            "legend": {
                "position": "top"
            },
            "title": {
                "display": true,
                "text": "Quarterly Sales Performance"
            }
            },
            "scales": {
            "y": {
                "beginAtZero": true
            }
            }
        }
        }
        """

    def _generate_visualization_data(self, query: str, summary: str) -> Dict[str, Any]:
        """
        Determines if a query/summary is suitable for visualization and, if so,
        generates the data in a Chart.js-compatible JSON format.

        Args:
            query: The original user query.
            summary: The text summary containing the data to be visualized.

        Returns:
            A dictionary formatted for Chart.js, or an empty dictionary {} if
            no visualization is generated or an error occurs.
        """
        # Step 1: Fast check to see if visualization is plausible.
        print(f"Checking if visualization is needed for query: '{query}'")
        try:
            check_prompt = (
                "Does the user query and the provided summary contain topics like data, trends, numbers, "
                "comparisons, or entities that would be suitable for a data visualization? "
                "Respond with only 'YES' or 'NO'.<br><br>"
                f"Query: '{query}'<br><br>Summary: '{summary}'"
            )
            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": check_prompt}],
                model=self.fast_model,
                max_tokens=5,
                temperature=0.0
            )
            decision = response.choices[0].message.content.strip().upper()
            print(f"AI decision for visualization: {decision}")
            if "YES" not in decision:
                return {}
        except Exception as e:
            print(f"Error in initial visualization check: {e}")
            return {}

        # Step 2: If plausible, generate the actual chart data.
        print("Visualization is needed. Generating chart data...")
        try:
            system_prompt = self._get_visualization_system_prompt()
            user_content = f"Query: '{query}'<br><br>Summary: '{summary}'"

            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model=self.fast_model,
                response_format={"type": "json_object"}
            )
            
            chart_json_string = response.choices[0].message.content
            
            # Step 3: Parse the JSON string and return the dictionary.
            return json.loads(chart_json_string)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from AI response: {e}")
            return {}
        except Exception as e:
            print(f"Error in visualization data generation: {e}")
            return {}

    def generate_summary(self, query: str) -> Dict[str, Any]:
        """
        Generates a long-form summary and conditionally adds visualization data.
        """
        user_prompt = f"Please perform a detailed analysis on the following topic: '{query}'"
        
        try:
            # 1. Generate the main summary as a JSON object
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=4096,
                # For models that support it, you can enforce JSON output
                # response_format={"type": "json_object"}, 
            )
            response_text = chat_completion.choices[0].message.content
            
            # Parse the AI's JSON response
            try:
                print(response_text)
                summary_markdown = response_text
            except json.JSONDecodeError:
                print(f"Failed to decode JSON from AI response")
                summary_markdown = "# Error<br><br>AI failed to return valid JSON. Please try again."

            # 2. Decide whether to add visualization data based on the query
            viz_data = self._generate_visualization_data(query, summary_markdown)
            return {"summary": summary_markdown, "visualization_data": viz_data}

        except Exception as e:
            print(f"An error occurred while generating the summary: {e}")
            error_message = f"# Error<br><br>Sorry, the summary could not be generated. **Details:** {e}"
            return {"summary": error_message, "visualization_data": {}}

