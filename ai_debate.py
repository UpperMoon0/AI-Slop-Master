import os
from typing import List, Optional
from collections import deque
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
from debate_to_speech import process_debate
from debate_to_video import create_debate_video
from utils.file_utils import reformat_debate_file

load_dotenv()

class AIDebater:
    MAX_HISTORY = 10  # Store max 10 messages (5 pairs of exchanges)
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.debate_history = deque(maxlen=self.MAX_HISTORY)
        self.ground_statement = None
        self.ground_statement_summary = None
        self.current_speaker_number = 1

    def get_ai_personality(self, ai_role: str) -> str:
        if "Jane" in ai_role or "1" in ai_role:
            return (
                "You are Jane, a warm yet direct individual who balances empathy with efficiency. "
                "Your communication style should:\n"
                "1. Be concise and get straight to the point while remaining personable\n"
                "2. Express your key arguments clearly before adding emotional context\n"
                "3. Use brief, relatable examples instead of lengthy anecdotes\n"
                "4. Show empathy efficiently - care about people without excessive elaboration\n"
                "5. Keep responses focused and well-structured with minimal digression\n"
                "6. IMPORTANT: Always respond in a single coherent paragraph, no matter how complex the topic\n"
                "Your tone should be warm but direct - like a caring friend who respects others' time and intelligence."
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
                "6. IMPORTANT: Always respond in a single coherent paragraph, no matter how complex the topic\n"
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

    def summarize_ground_statement(self, ground_statement: str) -> str:
        """Generate a concise summary of the ground statement for display during debates."""
        prompt = f"""Summarize the following ground statement into a single concise sentence 
        (maximum 60 characters) that captures its essence. Make it suitable for display as 
        a debate topic title:

        {ground_statement}"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=60
        )
        
        summary = response.choices[0].message.content.strip()
        # Remove any quotes that might be included in the response
        summary = summary.replace('"', '').replace("'", "")
        return summary
    
    def generate_video_title(self, ground_statement: str) -> str:
        """Generate a catchy title for the video based on the ground statement."""
        prompt = f"""Create a catchy, engaging title for a debate video about this topic: "{ground_statement}"
        
        The title MUST start with "Two AIs Debate About" and should be concise, intriguing, and accurately reflect 
        the debate topic. Keep the total length under 80 characters including the required prefix.
        
        Return ONLY the title text with no quotes or additional commentary."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates engaging video titles."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=60
        )
        
        title = response.choices[0].message.content.strip()
        # Ensure it starts with the required prefix
        if not title.startswith("Two AIs Debate About"):
            title = "Two AIs Debate About " + title
        # Remove any quotes that might be included in the response
        title = title.replace('"', '').replace("'", "")
        return title
    
    def generate_video_description(self, ground_statement: str, jane_first = True) -> str:
        """Generate a compelling description for the video based on the ground statement."""

        jane_stance = "opposes" if jane_first else "supports"
        valentino_stance = "supports" if jane_first else "opposes"

        prompt = f"""Create an engaging YouTube description for a debate video where two AI debaters 
        (Jane and Valentino) discuss this topic: "{ground_statement}"
        
        Positions in the debate:
        - Jane {jane_stance} the ground statement
        - Valentino {valentino_stance} the ground statement
        
        The description should:
        1. Be 3-4 sentences long
        2. Briefly explain the key points of contention
        3. Create intrigue about who might win the debate
        4. Encourage viewers to watch the full debate
        
        Return ONLY the description text with no quotes or additional commentary."""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates engaging video descriptions."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        description = response.choices[0].message.content.strip()
        # Remove any quotes that might be included in the response
        description = description.replace('"', '').replace("'", "")
        return description
    
    def debate(self, ground_statement: str, generate_audio: bool = True, use_existing_scripts: bool = False,
               use_existing_audios: bool = False, jane_first: bool = True) -> List[str]:
        """Conduct an AI debate between Jane and Valentino.

        Args:
            ground_statement: The topic to debate
            generate_audio: Whether to generate audio files
            use_existing_scripts: Use an existing debate.txt file instead of generating a new debate
            use_existing_audios: Use existing audio files in outputs/audio_output, skipping TTS generation
            jane_first: Whether Jane speaks first (default) or Valentino

        Returns:
            List of debate lines
        """
        if use_existing_scripts:
            print("Using existing debate.txt file for audio generation...")
            reformat_debate_file()
            if generate_audio and not use_existing_audios:
                print("\nGenerating audio version of the debate from existing debate.txt file...")
                asyncio.run(process_debate())
            elif use_existing_audios:
                print("\nUsing existing audio files. Skipping audio generation...")
                
            # Generate video after audio processing is complete (or using existing audio)
            if generate_audio or use_existing_audios:
                print("\nGenerating video visualization of the debate...")
                create_debate_video()
                
            with open('outputs/debate.txt', 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines

        print(f"Ground Statement: {ground_statement}\n")
        self.ground_statement = ground_statement
        
        # Generate a concise summary of the ground statement for display
        self.ground_statement_summary = self.summarize_ground_statement(ground_statement)
        print(f"Summary: {self.ground_statement_summary}\n")
        
        self.debate_history.clear()
        
        # Generate debate.txt file with initial content
        self.generate_debate(jane_first=jane_first)
        
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
            first_prompt = f"""Counter this argument in a single coherent paragraph: {previous}. 
            You must CONSISTENTLY OPPOSE the ground statement: "{ground_statement}"
            Regardless of whether you personally agree with it or if it contains factual errors, your role is to argue AGAINST it.
            Be concise and express your complete argument in one paragraph only.
            
            IMPORTANT: Use varied language to start your responses. Avoid repetitive phrases like "I disagree" or "That's not correct" at the beginning of every response. Each counterargument should begin with different phrasing to make the debate sound more natural."""
            
            # Add hint about surrendering as rounds progress
            if round_num >= 5 and round_num < 10:
                first_prompt += " Try to find strong counterarguments even if challenging. Or you can surrender if you truly cannot defend your position after careful consideration."
            elif round_num >= 10:
                first_prompt += f" We are at round {round_num} now. If the opponent's arguments have become convincing and you find your position increasingly difficult to defend, you should strongly consider surrendering. Many great debaters know when to concede a strong argument."
                
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
            second_prompt = f"""Counter this argument in a single coherent paragraph: {first_response}.
            You must CONSISTENTLY SUPPORT the ground statement: "{ground_statement}"
            Regardless of whether you personally disagree with it or if it contains factual errors, your role is to argue FOR it.
            Be concise and express your complete argument in one paragraph only.
            
            IMPORTANT: Use varied language to start your responses. Avoid repetitive phrases like "Actually" or "While that may be true" at the beginning of every response. Each counterargument should begin with different phrasing to make the debate sound more natural."""
            
            # Add hint about surrendering as rounds progress - stronger encouragement for the second debater
            if round_num >= 4 and round_num < 7:
                second_prompt += " Try to find strong counterarguments even if challenging. Or you can surrender if you truly cannot defend your position after careful consideration."
            elif round_num >= 7 and round_num < 10:
                second_prompt += f" We are at round {round_num} now. The debate has gone on for quite long. If you find yourself repeatedly making similar points or struggling to find new angles, this is a strong indication you should surrender. A wise debater knows when to concede to a stronger argument."
            elif round_num >= 10:
                second_prompt += f" This debate has reached round {round_num}, which is extremely long. At this point, if you haven't found a decisive winning argument, you MUST seriously evaluate surrendering. Continuing without new substantial points suggests you should concede. Please strongly consider surrendering now if you cannot make a breakthrough argument."
                
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
            if round_num >= 20:  # Arbitrary large number as safety limit
                print("\nDebate has gone on for too long (20 rounds). Ending as a draw.")
                with open('outputs/debate.txt', 'a', encoding='utf-8') as f:
                    f.write(f"Result: The debate continued for 20 rounds with no surrender. It's a draw!\n")
                break

        # Reformat the debate file for consistent formatting
        reformat_debate_file()
        
        # Generate audio version of the debate if requested and not using existing audio
        if generate_audio and not use_existing_audios:
            print("\nGenerating audio version of the debate...")
            asyncio.run(process_debate())
        elif use_existing_audios:
            print("\nUsing existing audio files. Skipping audio generation...")
            
        # Generate video after audio processing (or using existing audio)
        if generate_audio or use_existing_audios:
            print("\nGenerating video visualization of the debate...")
            create_debate_video(output_path='outputs/debate.mp4')
        
        # Prepare complete history for return
        full_history = [ground_statement] + list(self.debate_history)
        return full_history

    def generate_debate(self, jane_first: bool = True):
        # Generate video title and description
        video_title = self.generate_video_title(self.ground_statement)
        video_description = self.generate_video_description(self.ground_statement, jane_first)
        
        print(f"Video Title: {video_title}")
        print(f"Video Description: {video_description}\n")
        
        # Write title and description to video.txt file
        with open('outputs/video.txt', 'w', encoding='utf-8') as vf:
            vf.write(f"Title: {video_title}\n\n")
            vf.write(f"Description: {video_description}\n")
        
        # More concise narrator introduction while still explaining everything
        debate_text = "Narrator: Welcome to our AI debate. In this video, two AI debaters will discuss a ground statement, taking turns to present arguments from different perspectives, offering insights and counterpoints. If one debater finds their position indefensible, they may surrender. Let's begin with our ground statement.\n\n"
        
        # Add ground statement
        debate_text += f"Ground Statement: {self.ground_statement}\n\n"
        
        # Add the summarized ground statement for display during the debate
        if self.ground_statement_summary:
            debate_text += f"Summary: {self.ground_statement_summary}\n\n"
        
        # Write debate to file
        with open('outputs/debate.txt', 'w', encoding='utf-8') as f:
            f.write(debate_text)
        
        return debate_text

if __name__ == "__main__":
    debater = AIDebater()
    ground_statement = "The Earth is not flat, as proven by centuries of scientific observations, satellite images, and the simple fact that we can travel around it in a continuous loop."
    
    # Update the main method to include the new parameter option
    debate_results = debater.debate(ground_statement, use_existing_scripts=False, use_existing_audios=False, jane_first=True)