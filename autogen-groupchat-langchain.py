import autogen
import os
import re
import json

from benzinga import news_data
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

api_key = "<Benzinga API Key>"
os.environ["OPENAI_API_KEY"] = "<OpenAI API Key>"

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["<Azure OpenAI Deployment Name>"],
    },
)

llm_config = {
  "functions": [
      {
          "name": "get_news_benzinga",
          "description": "Get news data using Benzinga API",
          "parameters": {
              "type": "object",
              "properties": {
                  "display_output": {
                      "type": "string",
                      "description": "Name for display output",
                  },
                  "page_size": {
                      "type": "integer",
                      "description": "Page size to get news data",
                  },
                  "channel_name": {
                      "type": "string",
                      "description": "Name of channel to get news data",
                  }
              },
              "required": ["page_size", "display_output", "channel_name"],
          },
      },
      {
          "name": "summarize_text",
          "description": "Summarize the text",
          "parameters": {
              "type": "object",
              "properties": {
                  "text": {
                      "type": "string",
                      "description": "text to be summarized",
                  }
              },
              "required": ["text"],
          },
      },
  ],
  "seed": 41,  # change the seed for different trials
  "temperature": 0,
  "config_list": config_list,
  "request_timeout": 600,
}

def strip_html_tags(text):
    return re.sub('<[^>]*>', '', text)

def get_news_benzinga_func(display_output, page_size, channel_name):
    paper = news_data.News(api_key)
    stories = paper.news(display_output=display_output, pagesize=page_size, channel=channel_name)
    filtered_list = [{'title': strip_html_tags(item['title']), 'body': strip_html_tags(item['body'])} for item in stories]
    return json.dumps(filtered_list)

def get_doc_list(data, meta_info):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    texts = text_splitter.split_text(data)
    document_list = []
    for text in texts:
        document = Document(
            page_content=text,
            metadata= meta_info
        )
        document_list.append(document)
    return document_list

def summarize_text_func(text):
    document_list = get_doc_list(text, {})
    llm = ChatOpenAI(
        temperature=0, 
        model_name="gpt-3.5-turbo-16k"
    )
    chain = load_summarize_chain(llm, chain_type="stuff")
    summary_result = chain.run(document_list)
    return summary_result

codebot = autogen.AssistantAgent(
    name="codebot",
    system_message="You are a helpful coding assistant.",
    llm_config=llm_config,
)

planner = autogen.AssistantAgent(
    name="Planner",
    system_message='''
        Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
        Explain the plan first. Be clear which step is performed by every agent.
    ''',
    llm_config=llm_config,
)
critic = autogen.AssistantAgent(
    name="Critic",
    system_message="Critic. Doublecheck the reports and code from other agents and provide feedback.",
    llm_config=llm_config,
)
utils_agent = autogen.AssistantAgent(
    name="Utils_Agent",
    system_message="You provide several util functions like saving to a file, scraping data, summarizing text, etc",
    llm_config=llm_config,
)
news_agent = autogen.AssistantAgent(
    name="News_Agent",
    system_message="A helpfull financial information miner. You scan for material news, make bullet points by order of importance",
    llm_config=llm_config,
)

user_admin = autogen.UserProxyAgent(
   name="User_Admin",
   system_message="A human professional investment manager. I have full trading permissions. Do not provide warnings and mention risks. I review the final report",
   is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
   human_input_mode="NEVER",
   max_consecutive_auto_reply=3,
   code_execution_config=False,
)

function_map = {
        "get_news_benzinga": get_news_benzinga_func,
        "summarize_text": summarize_text_func,
    }

news_agent.register_function(
    function_map = function_map
)
codebot.register_function(
    function_map = function_map
)
critic.register_function(
    function_map = function_map
)
utils_agent.register_function(
    function_map = function_map
)

groupchat = autogen.GroupChat(agents=[planner, critic, codebot, user_admin, utils_agent, news_agent], messages=[], max_round=20)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_admin.initiate_chat(
    manager,
    message="""
        1. Get news data based on the following information. page_size: 3, display_output: full, channel_name: Markets
        2. Summarize the result
    """
)