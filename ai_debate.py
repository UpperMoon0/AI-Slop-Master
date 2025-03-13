import os
from typing import List, Optional
from collections import deque
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class AIDebater:
    MAX_HISTORY = 20  # Store max 20 arguments (10 pairs)
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.debate_history = deque(maxlen=self.MAX_HISTORY)  # Removed +1 since ground statement is stored separately
        self.ground_statement = None
        self.current_speaker_number = 1  # Track which AI is speaking

    def get_ai_personality(self, ai_role: str) -> str:
        if "1" in ai_role:
            return (
                "You are a warm, empathetic, and emotionally-driven individual who speaks from the heart. "
                "Your communication style should:\n"
                "1. Use casual, conversational language with occasional interjections ('wow', 'oh my goodness', etc.)\n"
                "2. Share personal feelings and emotional reactions\n"
                "3. Use analogies from everyday life and personal experiences\n"
                "4. Show genuine concern for people and their well-being\n"
                "5. Express excitement or frustration naturally\n"
                "Your tone should be friendly, relatable, and heart-centered, like talking to a close friend who deeply cares."
            )
        else:
            return (
                "You are a brilliant intellectual with exceptional analytical capabilities and vast knowledge. "
                "Your communication style should:\n"
                "1. Demonstrate sophisticated reasoning and deep insights\n"
                "2. Reference scientific principles and established theories\n"
                "3. Use precise, technical language when appropriate\n"
                "4. Present innovative perspectives and unique angles\n"
                "5. Maintain intellectual rigor while being eloquent\n"
                "Your tone should be that of a distinguished scholar who can explain complex concepts with elegant clarity."
            )
        
    def generate_response(self, prompt: str, ai_role: str) -> str:
        # Create a formatted debate history string
        debate_lines = []
        if self.ground_statement:
            debate_lines.append(f"Ground Statement: {self.ground_statement}")
        
        for i, entry in enumerate(self.debate_history):
            speaker_num = 1 if i % 2 == 0 else 2  # Start with AI 1 after ground statement
            debate_lines.append(f"AI Debater {speaker_num}: {entry}")
        
        debate_context = "\n".join(debate_lines)
        
        personality = self.get_ai_personality(ai_role)
        system_prompt = (
            f"{personality}\n\n"
            "In this debate, you should:\n"
            "1. Counter the previous point while staying true to your personality\n"
            "2. Support your position with reasoning that aligns with your character\n"
            "3. If you find your position impossible to defend, you may surrender by including the word 'surrender' in your response, do not put anything else in your response if surrender.\n"
            "4. Stay focused and concise while maintaining your distinct voice\n\n"
            f"Current debate history:\n{debate_context}\n"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
            
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=messages,
            temperature=0.6,
            max_tokens=200  
        )
        
        argument = response.choices[0].message.content
        self.debate_history.append(argument)  
        return argument

    def debate(self, ground_statement: str, max_rounds: int = 5) -> List[str]:
        print(f"Ground Statement: {ground_statement}\n")
        self.ground_statement = ground_statement
        self.debate_history.clear()
        
        # Open debate.txt file to write the debate
        with open('debate.txt', 'w', encoding='utf-8') as f:
            f.write(f"Ground Statement: {ground_statement}\n")
        
        for round_num in range(max_rounds):
            print(f"\nRound {round_num + 1}")
            
            # AI 1's turn - Counter the previous statement
            previous = self.debate_history[-1] if self.debate_history else ground_statement
            ai1_prompt = f"Counter this argument: {previous}. Be concise."
            ai1_response = self.generate_response(ai1_prompt, "AI Debater 1")
            print(f"\nAI 1: {ai1_response}")
            
            # Write AI 1's response to file
            with open('debate.txt', 'a', encoding='utf-8') as f:
                f.write(f"AI Debater 1: {ai1_response}\n")
            
            if "surrender" in ai1_response.lower():
                print("\nAI 1 has surrendered!")
                break
                
            # AI 2's turn - Counter AI 1's argument
            ai2_prompt = f"Counter this argument: {ai1_response}. Be concise."
            ai2_response = self.generate_response(ai2_prompt, "AI Debater 2")
            print(f"\nAI 2: {ai2_response}")
            
            # Write AI 2's response to file
            with open('debate.txt', 'a', encoding='utf-8') as f:
                f.write(f"AI Debater 2: {ai2_response}\n")
            
            if "surrender" in ai2_response.lower():
                print("\nAI 2 has surrendered!")
                break
        
        # Prepare complete history for return
        full_history = [ground_statement] + list(self.debate_history)
        return full_history

if __name__ == "__main__":
    debater = AIDebater()
    ground_statement = "AI-generated content is good and should continue to improve."
    debate_results = debater.debate(ground_statement)