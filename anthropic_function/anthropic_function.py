from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import json
import ast


class AnthropicFunction:

  def __init__(self,
               api_key,
               model="claude-2",
               temperature=0.5,
               max_tokens_to_sample=300):
    self.anthropic = Anthropic(api_key=api_key)
    self.model = model
    self.temperature = temperature
    self.max_tokens_to_sample = max_tokens_to_sample
    self.functions = {}

  def add_function(self, name, description, parameters):
    self.functions[name] = {
        "description": description,
        "parameters": parameters
    }

  def call(self,
           messages,
           model=None,
           temperature=None,
           max_tokens_to_sample=None):
    # Use the parameters if provided, otherwise use the instance variables
    model = model if model is not None else self.model
    temperature = temperature if temperature is not None else self.temperature
    max_tokens_to_sample = max_tokens_to_sample if max_tokens_to_sample is not None else self.max_tokens_to_sample

    function_calling_documentation = """Function calling
In an API call, you can describe functions to gpt-3.5-turbo-0613 and gpt-4-0613, and have the model intelligently choose to output a JSON object containing arguments to call those functions. The Chat Completions API does not call the function; instead, the model generates JSON that you can use to call the function in your code.

The latest models (gpt-3.5-turbo-0613 and gpt-4-0613) have been fine-tuned to both detect when a function should to be called (depending on the input) and to respond with JSON that adheres to the function signature. With this capability also comes potential risks. We strongly recommend building in user confirmation flows before taking actions that impact the world on behalf of users (sending an email, posting something online, making a purchase, etc).

Under the hood, functions are injected into the system message in a syntax the model has been trained on. This means functions count against the model's context limit and are billed as input tokens. If running into context limits, we suggest limiting the number of functions or the length of documentation you provide for function parameters.
Function calling allows you to more reliably get structured data back from the model. For example, you can:

Create chatbots that answer questions by calling external APIs (e.g. like ChatGPT Plugins)
e.g. define functions like send_email(to: string, body: string), or get_current_weather(location: string, unit: 'celsius' | 'fahrenheit')
Convert natural language into API calls
e.g. convert "Who are my top customers?" to get_customers(min_revenue: int, created_before: string, limit: int) and call your internal API
Extract structured data from text
e.g. define a function called extract_data(name: string, birthday: string), or sql_query(query: string)
...and much more!

The basic sequence of steps for function calling is as follows:

Call the model with the user query and a set of functions defined in the functions parameter.
The model can choose to call a function; if so, the content will be a stringified JSON object adhering to your custom schema (note: the model may generate invalid JSON or hallucinate parameters).
Parse the string into JSON in your code, and call your function with the provided arguments if they exist.
Call the model again by appending the function response as a new message, and let the model summarize the results back to the user.
You can see these steps in action through the example below:

import openai
import json


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    # Get the current weather in a given location"
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


def run_conversation():
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": "What's the weather like in Boston?"}]
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = fuction_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
        )  # get a new response from GPT where it can see the function response
        return second_response


print(run_conversation())
Hallucinated outputs in function calls can often be mitigated with a system message. For example, if you find that a model is generating function calls with functions that weren't provided to it, try using a system message that says: "Only use the functions you have been provided with."
In the example above, we sent the function response back to the model and let it decide the next step. It responded with a user-facing message which was telling the user the temperature in Boston, but depending on the query, it may choose to call a function again.

For example, if you ask the model “Find the weather in Boston this weekend, book dinner for two on Saturday, and update my calendar” and provide the corresponding functions for these queries, it may choose to call them back to back and only at the end create a user-facing message.

If you want to force the model to call a specific function you can do so by setting function_call: {"name": "<insert-function-name>"}. You can also force the model to generate a user-facing message by setting function_call: "none". Note that the default behavior (function_call: "auto") is for the model to decide on its own whether to call a function and if so which function to call.

You can find more examples of function calling in the OpenAI cookbook:

Function calling
Learn from more examples demonstrating function calling
Completions API Legacy
The completions API endpoint received its final update in July 2023 and has a different interface than the new chat completions endpoint. Instead of the input being a list of messages, the input is a freeform text string called a prompt.

An example API call looks as follows:

import openai

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Write a tagline for an ice cream shop."
)
See the full API reference documentation to learn more.

Token log probabilities
The completions API can provide a limited number of log probabilities associated with the most likely tokens for each output token. This feature is controlled by using the logprobs field. This can be useful in some cases to assess the confidence of the model in its output.

Inserting text
The completions endpoint also supports inserting text by providing a suffix in addition to the standard prompt which is treated as a prefix. This need naturally arises when writing long-form text, transitioning between paragraphs, following an outline, or guiding the model towards an ending. This also works on code, and can be used to insert in the middle of a function or file.

DEEP DIVE
Inserting text
Completions response format
An example completions API response looks as follows:

{
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "logprobs": null,
      "text": "\n\n\"Let Your Sweet Tooth Run Wild at Our Creamy Ice Cream Shack"
    }
  ],
  "created": 1683130927,
  "id": "cmpl-7C9Wxi9Du4j1lQjdjhxBlO22M61LD",
  "model": "text-davinci-003",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 16,
    "prompt_tokens": 10,
    "total_tokens": 26
  }
}
In Python, the output can be extracted with response['choices'][0]['text'].

The response format is similar to the response format of the chat completions API but also includes the optional field logprobs.

Chat Completions vs. Completions
The chat completions format can be made similar to the completions format by constructing a request using a single user message. For example, one can translate from English to French with the following completions prompt:

Translate the following English text to French: "{text}"
And an equivalent chat prompt would be:

[{"role": "user", "content": 'Translate the following English text to French: "{text}"'}]
Likewise, the completions API can be used to simulate a chat between a user and an assistant by formatting the input accordingly.

The difference between these APIs derives mainly from the underlying GPT models that are available in each. The chat completions API is the interface to our most capable model (gpt-4), and our most cost effective model (gpt-3.5-turbo). For reference, gpt-3.5-turbo performs at a similar capability level to text-davinci-003 but at 10% the price per token! See pricing details here.

Which model should I use?
We generally recommend that you use either gpt-4 or gpt-3.5-turbo. Which of these you should use depends on the complexity of the tasks you are using the models for. gpt-4 generally performs better on a wide range of evaluations. In particular, gpt-4 is more capable at carefully following complex instructions. By contrast gpt-3.5-turbo is more likely to follow just one part of a complex multi-part instruction. gpt-4 is less likely than gpt-3.5-turbo to make up information, a behavior known as "hallucination". gpt-4 also has a larger context window with a maximum size of 8,192 tokens compared to 4,096 tokens for gpt-3.5-turbo. However, gpt-3.5-turbo returns outputs with lower latency and costs much less per token.

We recommend experimenting in the playground to investigate which models provide the best price performance trade-off for your usage. A common design pattern is to use several distinct query types which are each dispatched to the model appropriate to handle them.

GPT best practices
An awareness of the best practices for working with GPTs can make a significant difference in application performance. The failure modes that GPTs exhibit and the ways of working around or correcting those failure modes are not always intuitive. There is a skill to working with GPTs which has come to be known as “prompt engineering”, but as the field has progressed its scope has outgrown merely engineering the prompt into engineering systems that use model queries as components. To learn more, read our guide on GPT best practices which covers methods to improve model reasoning, reduce the likelihood of model hallucinations, and more. You can also find many useful resources including code samples in the OpenAI Cookbook.

Managing tokens
Language models read and write text in chunks called tokens. In English, a token can be as short as one character or as long as one word (e.g., a or apple), and in some languages tokens can be even shorter than one character or even longer than one word.

For example, the string "ChatGPT is great!" is encoded into six tokens: ["Chat", "G", "PT", " is", " great", "!"].

The total number of tokens in an API call affects:

How much your API call costs, as you pay per token
How long your API call takes, as writing more tokens takes more time
Whether your API call works at all, as total tokens must be below the model’s maximum limit (4096 tokens for gpt-3.5-turbo)
Both input and output tokens count toward these quantities. For example, if your API call used 10 tokens in the message input and you received 20 tokens in the message output, you would be billed for 30 tokens. Note however that for some models the price per token is different for tokens in the input vs. the output (see the pricing page for more information).

To see how many tokens are used by an API call, check the usage field in the API response (e.g., response['usage']['total_tokens']).

Chat models like gpt-3.5-turbo and gpt-4 use tokens in the same way as the models available in the completions API, but because of their message-based formatting, it's more difficult to count how many tokens will be used by a conversation.

DEEP DIVE
Counting tokens for chat API calls
To see how many tokens are in a text string without making an API call, use OpenAI’s tiktoken Python library. Example code can be found in the OpenAI Cookbook’s guide on how to count tokens with tiktoken.

Each message passed to the API consumes the number of tokens in the content, role, and other fields, plus a few extra for behind-the-scenes formatting. This may change slightly in the future.

If a conversation has too many tokens to fit within a model’s maximum limit (e.g., more than 4096 tokens for gpt-3.5-turbo), you will have to truncate, omit, or otherwise shrink your text until it fits. Beware that if a message is removed from the messages input, the model will lose all knowledge of it.

Note that very long conversations are more likely to receive incomplete replies. For example, a gpt-3.5-turbo conversation that is 4090 tokens long will have its reply cut off after just 6 tokens.

FAQ
Why are model outputs inconsistent?
The API is non-deterministic by default. This means that you might get a slightly different completion every time you call it, even if your prompt stays the same. Setting temperature to 0 will make the outputs mostly deterministic, but a small amount of variability will remain.

How should I set the temperature parameter?
Lower values for temperature result in more consistent outputs, while higher values generate more diverse and creative results. Select a temperature value based on the desired trade-off between coherence and creativity for your specific application.

Is fine-tuning available for the latest models?
# No. Currently, you can only fine-tune base GPT-3 models (davinci, curie, babbage, and ada). See the fine-tuning guide for more details on how to use fine-tuned models.

Do you store the data that is passed into the API?
As of March 1st, 2023, we retain your API data for 30 days but no longer use your data sent via the API to improve our models. Learn more in our data usage policy. Some endpoints offer zero retention.

How can I make my application more safe?
If you want to add a moderation layer to the outputs of the Chat API, you can follow our moderation guide to prevent content that violates OpenAI’s usage policies from being shown.

Should I use ChatGPT or the API?
ChatGPT offers a chat interface to the models in the OpenAI API and a range of built-in features such as integrated browsing, code execution, plugins, and more. By contrast, using OpenAI’s API provides more flexibility."""
    functions = "\n".join([
        f"{idx+1}. {name}({', '.join(params)}): {desc}"
        for idx, (name, value) in enumerate(self.functions.items())
        for desc, params in [value.values()]
    ])

    prompt_prefix = f"""\n\nHuman: Here is the documentation for how you should call functions:\n```\n{function_calling_documentation}\n```\n\nHere are the available functions:\n```{functions}\n```\nRemember, you should output `FUNCTION $function_name($arguments)` when you want to call a function.
        
If you are calling a function, only include the function call -- no other text. For example, this is wrong.
```
FUNCTION get_current_weather(location: \'Boston\', unit: \'fahrenheit\')\n\nI have called the get_current_weather function to retrieve the weather in Boston.
```

Why is it wrong? It is wrong because the response included extra text that is not part of the function call. `\n\nI have called the get_current_weather function to retrieve the weather in Boston.`

This is what a correct function call output would look like:
```
FUNCTION get_current_weather(location: \'Boston\', unit: \'fahrenheit\')
```

If no function call is needed, converse like you would otherwise."""

    conversation = ""
    for message in messages:
      conversation += f'\n\n{message["role"].capitalize()}: {message["content"]}'

    prompt = f"{prompt_prefix}{conversation}\n\nAssistant:"

    response = self.anthropic.completions.create(
        model=model,
        max_tokens_to_sample=max_tokens_to_sample,
        prompt=prompt,
        temperature=temperature,
    )

    output = response.completion.strip()

    print(output)

    if output.startswith("FUNCTION "):
      # It's trying to call a function, so we parse the function name and arguments
      function_call = output[len("FUNCTION "):]
      return {"function": function_call}
        
    else:
      return {"response": output, "function": None, "arguments": None}

  def describe_function_output(self, function_name, function_arguments,
                               function_output, previous_messages):
    # Convert the function output and arguments to a string if they're not already
    if not isinstance(function_output, str):
      function_output = str(function_output)
    if not isinstance(function_arguments, str):
      function_arguments = str(function_arguments)

    # Add a new system message to the list of messages
    new_message = {
        "role":
        "AI",
        "content":
        f"The function `{function_name}` was called with the following arguments: `{function_arguments}`. It returned the following output: `{function_output}`. Tell the user what happened, in natural language. Don't include details about the code, just report back like a friend."
    }
    previous_messages.append(new_message)

    # Call the model again
    response = self.call(previous_messages)

    return response
