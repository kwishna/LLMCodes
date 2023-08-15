"""Chain that carries on a conversation and calls an LLM."""
from typing import Dict, List

from pydantic import Extra, Field, root_validator

from langchain.chains.conversation.prompt import PROMPT
from langchain.chains.llm import LLMChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseMemory


class ConversationChain(LLMChain):
    """Chain to have a conversation and load context from memory.

    Example:
        .. code-block:: python

            from langchain import ConversationChain, OpenAI
            conversation = ConversationChain(llm=OpenAI())
    """

    memory: BaseMemory = Field(default_factory=ConversationBufferMemory)
    """Default memory store."""
    prompt: BasePromptTemplate = PROMPT
    """Default conversation prompt to use."""

    input_key: str = "input"  #: :meta private:
    output_key: str = "response"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Use this since so some prompt vars come from history."""
        return [self.input_key]

    def __init__(self, **kwargs):
        """Initialize the chain."""
        super().__init__(**kwargs)
        print("here are my kwargs")
        for key in kwargs:
            print(key)
            if key=="llm":
                print(list(kwargs.values())[0].openai_api_key)
        print("here are my kwargs")
        if self.memory is not None and self.memory.input_key is None:
            self.memory.input_key = self.input_key

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        """Validate that prompt input variables are consistent."""
        memory_keys = values["memory"].memory_variables
        input_key = values["input_key"]
        if input_key in memory_keys:
            raise ValueError(
                f"The input key {input_key} was also found in the memory keys "
                f"({memory_keys}) - please provide keys that don't overlap."
            )
        prompt_variables = values["prompt"].input_variables
        expected_keys = memory_keys + [input_key]
        if set(expected_keys).difference(set(prompt_variables)):
            raise ValueError(
                "Prompt input variables do not contain all expected keys. Expected keys are "
                f"{expected_keys}, but got {prompt_variables} from the prompt."
            )
        print("here are my values")
        print(values)
        print("here are my values")
        return values