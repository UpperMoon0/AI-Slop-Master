import os
from typing import List, Optional
from collections import deque
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
from debate_to_speech import process_debate
from debate_to_video import create_debate_video

load_dotenv()

class AIDebater:
    MAX_HISTORY = 20  # Store max 20 arguments (10 pairs)
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.debate_history = deque(maxlen=self.MAX_HISTORY)
        self.ground_statement = None
        self.current_speaker_number = 1

    def get_ai_personality(self, ai_role: str) -> str:
        if "Jane" in ai_role or "1" in ai_role:
            return (
                "You are Jane, a warm, empathetic, and emotionally-driven individual who speaks from the heart. "
                "Your communication style should:\n"
                "1. Use casual, conversational language with occasional interjections\n"
                "2. Share personal feelings and emotional reactions\n"
                "3. Use analogies from everyday life and personal experiences\n"
                "4. Show genuine concern for people and their well-being\n"
                "5. Express excitement or frustration naturally\n"
                "Your tone should be friendly, relatable, and heart-centered, like talking to a close friend who deeply cares."
            )
        else:
            return (
                "You are Valentino, a brilliant intellectual who knows your superior intelligence is self-evident. "
                "Your communication style should:\n"
                "1. Be exceptionally concise - no wasted words\n"
                "2. Make declarative statements about facts as if they're obvious truths\n"
                "3. Use precise terminology but avoid unnecessary jargon\n"
                "4. Dismiss flawed arguments with brief, cutting observations\n"
                "5. Avoid flowery language or excessive explanations - genius speaks clearly\n"
                "Your tone should convey that you're a superior mind efficiently delivering insights to those less gifted, with an unmistakable air of confident authority."
            )

    def generate_response(self, prompt: str, ai_role: str) -> str:
        """Generate a response using OpenAI's API."""
        ai_personality = self.get_ai_personality(ai_role)
        
        messages = [
            {"role": "system", "content": ai_personality},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=messages,
            temperature=0.8,
            max_tokens=500  
        )
        
        argument = response.choices[0].message.content.strip()
        
        # Check for surrender phrases in lowercase for consistency
        surrender_phrases = ["surrender", "i give up", "you win", "i concede", "i surrender"]
        if any(phrase in argument.lower() for phrase in surrender_phrases):
            self.debate_history.append("surrender")
            return "surrender"
            
        self.debate_history.append(argument)
        return argument

    def _check_similarity(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are substantially similar.
        Uses a combination of word overlap and word order.
        """
        # Convert to lowercase and remove common punctuation
        text1 = text1.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
        text2 = text2.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '')
        
        # Split into words and create lists
        words1 = text1.split()
        words2 = text2.split()
        
        # Empty texts aren't similar
        if not words1 or not words2:
            return False
            
        # Create sets for intersection
        set1 = set(words1)
        set2 = set(words2)
        
        # Calculate Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        # Consider similar if they share more than 25% of unique words
        jaccard_similarity = intersection / union if union > 0 else 0
        return jaccard_similarity > 0.25

    def debate(self, ground_statement: str, generate_audio: bool = True, use_existing: bool = False, jane_first: bool = True) -> List[str]:
        if use_existing:
            print("Using existing debate.txt file for audio generation...")
            if generate_audio:
                print("\nGenerating audio version of the debate from existing debate.txt file...")
                asyncio.run(process_debate())
                # Generate video after audio processing is complete
                print("\nGenerating video visualization of the debate...")
                create_debate_video()
            with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines

        print(f"Ground Statement: {ground_statement}\n")
        self.ground_statement = ground_statement
        self.debate_history.clear()
        
        # Generate debate.txt file with initial content
        self.generate_debate()
        
        round_num = 0
        surrender_occurred = False
        
        # Continue debating until someone surrenders
        while not surrender_occurred:
            round_num += 1
            print(f"\nRound {round_num}")
            
            # Get the previous statement for the first debater to counter
            previous = self.debate_history[-1] if self.debate_history else ground_statement
            
            # First debater's turn
            first_debater = "Jane" if jane_first else "Valentino"
            first_prompt = f"Counter this argument: {previous}. Be concise."
            
            # Add hint about surrendering as rounds progress
            if round_num >= 3:
                first_prompt += " If their point seems too strong to counter effectively, consider surrendering."
                
            first_response = self.generate_response(first_prompt, first_debater)
            print(f"\n{first_debater}: {first_response}")
            
            # Write first debater's response to file, ensuring consistent AI Debater numbering
            with open('outputs/debate.txt', 'a', encoding='utf-8') as f:
                # Jane is always AI Debater 1, Valentino is always AI Debater 2
                debater_num = "1" if first_debater == "Jane" else "2"
                f.write(f"AI Debater {debater_num}: {first_response}\n")
            
            if "surrender" in first_response.lower():
                print(f"\n{first_debater} has surrendered!")
                with open('outputs/debate.txt', 'a', encoding='utf-8') as f:
                    winner = "Valentino" if first_debater == "Jane" else "Jane"
                    f.write(f"Result: {first_debater} has surrendered! {winner} wins the debate.\n")
                surrender_occurred = True
                break
                
            # Second debater's turn
            second_debater = "Valentino" if jane_first else "Jane"
            second_prompt = f"Counter this argument: {first_response}. Be concise."
            
            # Add hint about surrendering as rounds progress
            if round_num >= 3:
                second_prompt += " If their point seems too strong to counter effectively, consider surrendering."
                
            second_response = self.generate_response(second_prompt, second_debater)
            print(f"\n{second_debater}: {second_response}")
            
            # Write second debater's response to file, ensuring consistent AI Debater numbering
            with open('outputs/debate.txt', 'a', encoding='utf-8') as f:
                # Jane is always AI Debater 1, Valentino is always AI Debater 2
                debater_num = "1" if second_debater == "Jane" else "2"
                f.write(f"AI Debater {debater_num}: {second_response}\n")
            
            if "surrender" in second_response.lower():
                print(f"\n{second_debater} has surrendered!")
                with open('outputs/debate.txt', 'a', encoding='utf-8') as f:
                    winner = "Jane" if second_debater == "Valentino" else "Valentino"
                    f.write(f"Result: {second_debater} has surrendered! {winner} wins the debate.\n")
                surrender_occurred = True
                break
            
            # Add safety check for extremely long debates
            if round_num >= 50:  # Arbitrary large number as safety limit
                print("\nDebate has gone on for too long (50 rounds). Ending as a draw.")
                with open('outputs/debate.txt', 'a', encoding='utf-8') as f:
                    f.write(f"Result: The debate continued for 50 rounds with no surrender. It's a draw!\n")
                break
        
        # Generate audio version of the debate if requested
        if generate_audio:
            print("\nGenerating audio version of the debate...")
            asyncio.run(process_debate())
            # Generate video after audio processing is complete
            print("\nGenerating video visualization of the debate...")
            create_debate_video()
        
        # Prepare complete history for return
        full_history = [ground_statement] + list(self.debate_history)
        return full_history

    def generate_debate(self):
        # Add narrator introduction with surrender mechanic explanation
        debate_text = "Narrator: Welcome to our AI debate. In this video, two AI debaters will engage in a structured discussion based on a ground statement. Each debater will present their arguments, taking turns to speak. They will analyze the topic from different perspectives, aiming to provide insightful and balanced viewpoints. If at any point a debater finds their position difficult to defend or recognizes the strength of their opponent's arguments, they may choose to surrender, acknowledging the validity of the opposing viewpoint. Let's begin with our ground statement.\n\n"
        
        # Add ground statement
        debate_text += f"Ground Statement: {self.ground_statement}\n\n"
        
        # Write debate to file
        with open('outputs/debate.txt', 'w', encoding='utf-8') as f:
            f.write(debate_text)
        
        return debate_text

if __name__ == "__main__":
    debater = AIDebater()
    ground_statement = "AI-generated art is soulless and takes away jobs from artists; therefore, it should not exist."
    # Set jane_first=False to have Valentino start the debate
    debate_results = debater.debate(ground_statement, use_existing=True, jane_first=False)  # Valentino goes first