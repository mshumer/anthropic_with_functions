# Anthropic with Functions

This library allows you to use the Anthropic Claude models with OpenAI-like Functions.

It's super rough and early, so feel free to make improvements if you want!

## Installation

You can install this package directly from GitHub:

```bash
pip install git+https://github.com/mshumer/anthropic_with_functions.git
```

## Usage

Here's a basic usage example:

```python
from anthropic_function import AnthropicFunction
import json

anthropic_func = AnthropicFunction(api_key="ANTHROPIC_API_KEY", model="claude-2", temperature=0.7, max_tokens_to_sample=500)

# Define your functions
def get_current_weather(location, unit="fahrenheit"):
  # Get the current weather in a given location
  weather_info = {
      "location": location,
      "temperature": "72", # hardcoded for the example
      "unit": unit,
      "forecast": ["sunny", "windy"], # hardcoded for the example
  }
  return json.dumps(weather_info)

# Add your functions to the AnthropicFunction instance
anthropic_func.add_function(
    "get_current_weather", "Get the current weather in a given location",
    ["location: string", "unit: 'celsius' | 'fahrenheit'"])

# Define the conversation messages
messages = [{"role": "HUMAN", "content": "how are you today?"}, {"role": "AI", "content": "I'm good, thanks for asking!"}, {"role": "HUMAN", "content": "Remind me what I just asked you?"}, {"role": "AI", "content": "You just asked me, how are you today? and I responded, I'm good, thanks for asking!"}, {"role": "HUMAN", "content": "What's the weather in London?"}]

# Call a function
response = anthropic_func.call(messages, model="claude-2", temperature=0.8, max_tokens_to_sample=400)

if response["function"]:
  # Parse and then call the function with the arguments
  function_output = None

  # Depending on your function(s), write parsing code to grab the function name and arguments
  #### PARSING CODE GOES HERE
  function_name = 'get_current_weather' # placeholder -- replace with your parsing code that grabs the function name
  function_arguments = {'location': 'london', 'unit': 'celsius'} # placeholder -- replace with your parsing code that grabs the function arguments

  # Now, call the relevant function with the arguments, return the result as `function_output`
  if function_name == 'get_current_weather':
    function_output = get_current_weather(location=function_arguments['location'], unit=function_arguments['unit'])
  # Describe the function's output
  if function_output is not None:
      response = anthropic_func.describe_function_output(function_name, function_arguments, function_output, messages)
      print('Response:', response['response'])

else:
  print('No function found')
  print('Response:', response['response'])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Some ideas:
- create automatic function / arguments parsing code so that the user doesn't need to write it themselves
- generally get the library to parity w/ OpenAI's Functions system

## License

This project is licensed under the terms of the MIT license.