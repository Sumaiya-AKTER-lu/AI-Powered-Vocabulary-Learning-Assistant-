import re
import sys
import io
from contextlib import redirect_stdout
from crewai import Crew, Process
from sem6.crew import VocabularyCrew
from sem6.tools.database_tools import DatabaseTools
from sem6.tools.custom_tools import SpeechToTextTool, GTTSTool

def extract_friendly_final_answer(output: str) -> str:
    matches = re.findall(r"## Final Answer:\s*(.*?)(?=\n## |🤖|$)", output, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return "Sorry, no friendly final answer was found."

def run():
    db = DatabaseTools()
    audio = SpeechToTextTool()
    tts = GTTSTool()

    print("🎓 Welcome to the Vocabulary Chatbot!")
    user_name = input("Enter your name: ")
    user_id = db.get_or_create_user_id(user_name)

    while True:
        print("\nWhat would you like to do?")
        print("1. Learn Vocabulary")
        print("2. Take a Quiz")
        print("3. Exit")
        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == '3':
            print("👋 Goodbye!")
            break

        crew_builder = VocabularyCrew()

        if choice == '1':
            print("\nWould you like to (1) Type or (2) Speak your query?")
            mode = input("Enter 1 or 2: ").strip()

            if mode == '2':
                print("🎤 Listening for your query...")
                try:
                    user_query = audio._run()
                    if not user_query.strip():
                        print("❗ I couldn't hear you clearly. Please try again.")
                        continue
                    print(f"You (spoken): {user_query}")
                except Exception as e:
                    print(f"[ERROR] Speech recognition failed: {e}")
                    continue
            else:
                user_query = input("You (typed): ")

            if user_query.lower() == 'exit':
                print("👋 Exiting...")
                break

            inputs = {'user_query': user_query}

            crew = Crew(
                agents=[crew_builder.extractor(), crew_builder.tutor()],
                tasks=[crew_builder.extractor_tasks(), crew_builder.tutor_tasks()],
                process=Process.sequential,
                verbose=True,
                shared_memory=True
            )

        elif choice == '2':
            user_query = input("📋 Enter the quiz topic or type 'last' for recent words: ")
            inputs = {'user_query': user_query}

            crew = Crew(
                agents=[crew_builder.quiz()],
                tasks=[crew_builder.quiz_tasks()],
                process=Process.sequential,
                verbose=True,
                shared_memory=True
            )

        else:
            print("❌ Invalid choice. Try again.")
            continue

        try:
            result = crew.kickoff(inputs=inputs)
            full_output = result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            print(f"[ERROR] Crew failed: {e}")
            continue

        final_answer = extract_friendly_final_answer(full_output)

        print("\n🧠 Chatbot Answer:\n" + final_answer)

        try:
            tts._run(final_answer)
        except Exception as e:
            print(f"[AUDIO ERROR] Text-to-speech failed: {e}")

        db.log_conversation(user_id, user_query, final_answer)


