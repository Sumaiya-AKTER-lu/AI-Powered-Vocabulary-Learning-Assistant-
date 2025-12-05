#crew.py

from typing import ClassVar, List
from pathlib import PosixPath

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from sem6.tools.custom_tools import GTTSTool

from sem6.tools.custom_tools import (
    WordnetExtractorTool,
    #OpenAIExtractorTool,
    RAGValidatorTool,
    TextToSpeechTool,
    SpeechToTextTool,
    GetProgressTool,
    FuzzyMatchTool,
    VocabularyPresenterTool,
    QuizGeneratorTool,
    GTTSTool
)

@CrewBase
class VocabularyCrew():
    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['extractor'],
            verbose=True,
            #verbose=False, 
            tools=[
                WordnetExtractorTool(),
                #OpenAIExtractorTool(),
                RAGValidatorTool()
            ]
        )

    @agent
    def tutor(self) -> Agent:
        return Agent(
            config=self.agents_config['tutor'],
            verbose=True,
            #verbose=False, 
            tools=[
                TextToSpeechTool(),
                SpeechToTextTool(),
                VocabularyPresenterTool(),
                GTTSTool()
            ]
        )

    @agent
    def quiz(self) -> Agent:
        return Agent(
            config=self.agents_config['quiz'],
            verbose=True,
            #verbose=False, 
            tools=[
                TextToSpeechTool(),
                SpeechToTextTool(),
                GetProgressTool(),
                FuzzyMatchTool(),
                QuizGeneratorTool(),
                GTTSTool()
            ]
        )

    @task
    def extractor_tasks(self) -> Task:
        return Task(
            config=self.tasks_config['extractor_task'],
            tools=[WordnetExtractorTool(), RAGValidatorTool()]
            #tools=[OpenAIExtractorTool(), RAGValidatorTool()]
        )

    @task
    def tutor_tasks(self) -> Task:
        return Task(
            config=self.tasks_config['tutor_task'],
            tools=[VocabularyPresenterTool(), GTTSTool()]
        )

    @task
    def quiz_tasks(self) -> Task:
        return Task(
            config=self.tasks_config['quiz_task'],
            tools=[GetProgressTool(), QuizGeneratorTool(), GTTSTool()]
        )






