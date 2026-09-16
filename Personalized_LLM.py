from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import gradio as gr

# Model: gpt-4o-mini | Provider: openai
model = init_chat_model("gpt-4o-mini", model_provider="openai")
agent_format_str = """
Your task is to explain the concept of **{agent}** to me in a way that is:

  1. Clear and intuitive
  2. Concise (in under 100 words)
  3. Tailored specifically to me and what I already know

  Use the following information about me to personalize your explanations:

  - Background: Junior IT Engineer and CS student at Boston University
  - Professional Interests: Software engineering, AI engineering, building LLM-powered applications
  - Personal Goals: Building an AI tutor to help with my university coursework
  - Learning Style: Prefer clear, practical explanations connected to real engineering problems

  The personalization should be subtle and natural. Avoid forced references to my background that don't genuinely enhance understanding.
  """
template_agent = PromptTemplate.from_template(agent_format_str)



def generate_text(agent_input):

  concept = agent_input
  prompt = template_agent.format(agent=concept)
  response = model.invoke(prompt)
  return response.text


demo = gr.Interface(
    fn=generate_text,
    inputs=[gr.Textbox(label="Enter a concept", lines=1)],
    outputs=[gr.Textbox(label="Explanation", lines=5)],
    flagging_mode="never",
    title="Study term explainer",
    description="Get help with your study plan for the term"
)
