from typing import TypedDict, Annotated, List, Union
import operator

class AgentState(TypedDict):
    question: str
    qid: str
    choices: List[str]
    category: str # 'math', 'rag', 'reading', 'toxic'
    context: str
    answer: str
    reasoning: str
