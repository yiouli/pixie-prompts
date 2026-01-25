# pixie-prompts
**Code-first, type-checked prompt management.**
Manage prompt locally in your codebase, with Jinja rendering, variable type-hint and validations.

[**Demo**](https://github.com/user-attachments/assets/aba55aca-1ad3-4f25-97f9-db0f8e67dbe6)

[**Try it live**](https://gopixie.ai/?url=https%3A%2F%2Fpixie-prompts-examples.vercel.app%2Fgraphql)

## Setup




In your project folder, install **pixie-prompts[server]** Python package:
```bash
pip install pixie-prompts[server]
```
> Note: you can install **pixie-prompts** without the server extras for your production build.

Start the local dev server and open the web UI by running:
```bash
pp
```
> Note: The web-browser would automatically open [http://localhost:8000](http://localhost:8000). You can also access the web UI at [gopixie.ai](https://gopixie.ai).

To test prompts, create *.env* file with LLM API key(s):
```ini
# .env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

## Register Prompt

In your code, create a new prompt using `create_prompt`:
```python
# prompts.py
from pixie.prompts import create_prompt

simple_prompt = create_prompt('simple_prompt')
```

Your prompt would automatically appear in the web UI after your code is saved.


## Manage Prompt

You can create new version(s) of a prompt in the web UI.

Once saved from web UI, it will be assigned a new version id, and the content would be saved in your codebase at */.pixie/prompts/<prompt_name>/<version_id>.jinja*.

> Note: it's recommended to only edit your prompts via the web UI to get type-hint and validation.


## Define Variables

For prompt that has variable(s) in it, define a class extending `pixie.prompts.Variables` (which extends `pydantic.BaseModel`. Then use the class type when registering your prompt.

```python
# prompts.py
from pixie.prompts import Variables, create_prompt

class Person(Variables):
    name: str
    age: int

# Create a prompt with variables
typed_prompt = create_prompt('typed_prompt', Person)
```

Other than using dict, you can define your variable class in anyway that's permissible in Pydantic. I.e. you can define your field as basic types such as `str`, `int`, `bool`, you can have a `list` of permissible items, you can use `Union` type, and you can have nested `Variable` field.

The web UI will parse your variable definitions and use it to decide input fields, type-hints and validations.


## Use Prompt

Compile your prompt into string with the `compile` function on the prompt object. Pass in the Variables object (if defined) for your prompt as argument.
```python
# demo.py

from pixie.prompts import Variables, create_prompt

simple_prompt = create_prompt('simple_prompt')

class Person(Variables):
    name: str
    age: int

# Create a prompt with variables
typed_prompt = create_prompt('typed_prompt', Person)

simple_prompt_str = simple_prompt.compile()
typed_prompt_str = typed_prompt.compile(Person(name="Jane", age=30))

```



Check out more [examples](https://github.com/yiouli/pixie-prompts-examples/blob/main/examples/prompts.py).
