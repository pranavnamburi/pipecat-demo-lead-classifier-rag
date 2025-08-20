# # --- 1. Imports ---
import os
import asyncio
from dotenv import load_dotenv
from twilio.rest import Client

# Your tool managers
from tools.csv_manager import CSVManager
from tools.rag_manager import RAGManager

# Custom transport
from twilio_transport import TwilioWebsocketTransport

# Pipecat services, processors, and frames
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.deepgram import DeepgramSTTService, LiveOptions
from pipecat.services.openai import OpenAILLMService, OpenAITTSService
from pipecat.frames.frames import Frame, TextFrame, InterimTranscriptionFrame, TranscriptionFrame, AudioRawFrame, EndFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

AudioRawFrame.name = "audio_raw"


# class TextAggregator(FrameProcessor):
#     """
#     Aggregates multiple downstream TextFrames into one, forwarding to the next processor
#     only after a sentence is complete (based on punctuation).
#     """
#     def __init__(self):
#         super().__init__()
#         self.buffer = []

#     async def process_frame(self, frame, direction):
#         await super().process_frame(frame, direction)

#         # Only aggregate downstream text frames
#         if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
#             self.buffer.append(frame.text)
#             # Heuristically flush when sentence seems complete
#             if frame.text.strip().endswith(('.', '।', '?', '!', '\n')):
#                 combined_text = ''.join(self.buffer).strip()
#                 if combined_text:
#                     new_frame = TextFrame(combined_text)
#                     await self.push_frame(new_frame, direction)
#                     self.buffer = []
#             # Else: do not push yet, keep buffering
#         else:
#             # Forward non-text or upstream frames immediately
#             await self.push_frame(frame, direction)

class FilterInterimFrames(FrameProcessor):
    """A simple processor to drop InterimTranscriptionFrames."""
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if not hasattr(frame, "text"):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, InterimTranscriptionFrame):
            # Drop the frame by not pushing it forward
            return
        # Forward all other frames
        await self.push_frame(frame, direction)

# class FilterTranscriptionFrames(FrameProcessor):
#     async def process_frame(self, frame, direction):
#         # Only pass TextFrame downstream; DROP all others silently (do NOT call super).
#         if not hasattr(frame, "text"):
#             await super().process_frame(frame, direction)
#             await self.push_frame(frame, direction)
#             return
#         if isinstance(frame, TranscriptionFrame) and direction == FrameDirection.DOWNSTREAM:
#             await self.push_frame(frame, direction)
#         else:
#             # Do NOT call await super().process_frame(frame, direction)
#             # Just drop all other frames
#             pass

    # Drop all other frames (or you could push upstream if needed)
class FilterAudioFrames(FrameProcessor):
    async def process_frame(self, frame, direction):
        # Drop all AudioRawFrame and pass all others

        if isinstance(frame, AudioRawFrame):
            # Do NOT call super, do NOT forward
            return
        if not hasattr(frame, "text"):
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return
        await self.push_frame(frame, direction)


# class DebugProcessor(FrameProcessor):
#     def __init__(self, debug_name: str):
#         super().__init__()
#         self.debug_name = debug_name

#     async def process_frame(self, frame, direction):
#         # await super().process_frame(frame, direction)
#         try:
#             if hasattr(frame, "text"):
#                 if isinstance(frame, TextFrame):
#                     print(f"{self.debug_name} [{type(frame).__name__}]: '{frame.text}' ({direction})")
#                 if isinstance(frame, TranscriptionFrame):
#                     print(f"{self.debug_name} [{type(frame).__name__}]: '{frame.text}' ({direction})")
#                 if isinstance(frame, InterimTranscriptionFrame):
#                     print(f"{self.debug_name} [{type(frame).__name__}]: '{frame.text}' ({direction})")
#                 await self.push_frame(frame, direction)
#             else:
#                 print(f"{self.debug_name} [{type(frame).__name__}]: '{frame}' ({direction})")
#         except Exception as e:
#             print(f"Error in DebugProcessor: {e}")
#             await self.push_frame(frame, direction)


# --- 2. Setup Function ---
def setup():
    """Initializes all necessary components."""
    load_dotenv()
    csv_manager = CSVManager(csv_path="data/leads.csv")
    rag_manager = RAGManager(csv_path="data/rag_source/kb_unresponsive.csv")
    twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))
    return csv_manager, rag_manager, twilio_client


class LeadClassificationProcessor(FrameProcessor):
    def __init__(self, csv_manager: CSVManager, rag_manager: RAGManager):
        super().__init__()
        self.csv_manager = csv_manager
        self.rag_manager = rag_manager
        self.conversation_history = []
        self.final_lead_status = "Other"
        self.final_call_summary = "Call completed without classification."
        self.call_ended = None
    
    async def handle_call_ended(self):
        self.call_ended = True
        print(f"!!! Call ended")
        await asyncio.sleep(8.0)
        await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        await self.push_frame(EndFrame(), FrameDirection.UPSTREAM)
        
        return

    async def handle_set_lead_status(self, params: FunctionCallParams)->str:
        """
        Sets the status of a lead.

        Args:
            status: The status to set for the lead (Interested/Not Interested/Other).
            summary: A one-sentence summary of the call.

        Returns:
            A user-facing Hindi message as a string That will be used to end the call.

        """
        status = params.arguments.get("status")
        summary = params.arguments.get("summary")
        self.final_lead_status = status
        self.final_call_summary = summary
        self.call_ended = True
        print(f"!!! LLM decided status: {status}, summary: {summary}")
        if status == "Interested":
            message = "बहुत बढ़िया! हमारी टीम जल्द ही आपसे संपर्क करेगी और सारी जानकारी देगी। धन्यवाद!"
        else:
            message = "बिल्कुल ठीक है। आपके समय के लिए धन्यवाद!"
        await params.result_callback({"message": message})
        await self.handle_call_ended()
        

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        print(f"[LeadClassificationProcessor] >>> Received frame: {type(frame).__name__}, direction: {direction}")

        # --- Only call super for system/lifecycle frames (frames without .text) ---
        if not hasattr(frame, "text"):
            print("[LeadClassificationProcessor] System/lifecycle frame, calling super and push.")
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)
            return

        # --- Handle user TranscriptionFrame (finalized speech input) ---
        if isinstance(frame, TextFrame) and direction == FrameDirection.DOWNSTREAM:
            print(f"[LeadClassificationProcessor] Handling user TranscriptionFrame with text='{frame.text}'")
            if self.call_ended:
                print("[LeadClassificationProcessor] Call ended, dropping frame.")
                return  # Don't process further on ended call

            self.conversation_history.append(f"User: {frame.text}")
            retrieved_context = self.rag_manager.search(frame.text)

            prompt = f"""You are a helpful lead classification assistant for a delivery job.

Context from knowledge base: "{retrieved_context}"

Your task:
1. Continue the conversation naturally based on the user's message.
2. When you have a clear answer about their interest, you MUST classify the lead by calling the 'handle_set_lead_status' function.
3. Use the available tool calls to call the function.You have this tool call available: handle_set_lead_status(status, summary)  is a function that takes in a status and a summary and returns a message.

Conversation so far:
---
{"\n".join(self.conversation_history)}
---
USER'S LATEST MESSAGE: "{frame.text}"

Your response:"""

            llm_input_frame = TranscriptionFrame(user_id=" ",text=prompt, timestamp=datetime.now(timezone.utc).isoformat(timespec='milliseconds'))
            print(f"[LeadClassificationProcessor] Pushing prompt (TextFrame) downstream: {prompt[:80]}...")
            await self.push_frame(llm_input_frame, direction)
            return

        

        # --- Any other .text frame, just pass through for safety ---
        print(f"[LeadClassificationProcessor] Unhandled .text frame, passing through: {type(frame).__name__}")
        await self.push_frame(frame, direction)

# --- 4. Main Application ---
async def main():
    csv_manager, rag_manager, twilio_client = setup()
    classification_processor = LeadClassificationProcessor(csv_manager, rag_manager)

    lead = csv_manager.get_next_lead()
    if not lead:
        print("No new leads to call. Exiting.")
        return

    print(f"Calling lead: {lead['name']} at {lead['phone_number']}")

    transport = TwilioWebsocketTransport(
            host="0.0.0.0",  # Allow external connections through ngrok
            port=8765
        )
    stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            sample_rate=16000,
            live_options=LiveOptions(
                model="nova-2",
                language="hi",
                smart_format=True,          # English for better recognition
                interim_results=False,  # Only final results to prevent duplicates
                encoding="linear16",
            )      # Voice activity detection
        )
    lead_status_tool = FunctionSchema(
            name="handle_set_lead_status",
            description="Sets the final lead status and summary for the call.",
            properties={
                "status": {"type": "string","description": "The status of the lead (Interested/Not Interested/Other)" ,"enum": ["Interested", "Not Interested", "Other"]},
                "summary": {"type": "string","description": "A one-sentence summary of the call."},
            },
            required=["status", "summary"],
        )

    tools=ToolsSchema(standard_tools=[lead_status_tool])
    llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            max_tokens=75,
            temperature=0.4,
            function_call="auto",
        )
    llm.register_function("handle_set_lead_status", classification_processor.handle_set_lead_status)

    tts = OpenAITTSService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini-tts", voice="nova")

    messages = [
        {
            "role": "system",
            "content": (
                "You are Priya (female) from वाहन (pronunced as वा हन, not वहाँ) Jobs currently on a phone call with a lead. "
                "You have DELIVERY jobs available at ALL LOCATIONS (don't talk about location until asked). Your main GOAL in the CALL is to check if the user is interested in a delivery job "
                "(keep in mind to try to get the user to say yes). Keep your outputs brief and to the point, natural, casual and humane as if you are a human on a call, "
                "so no special characters and your inner thinking and don't repeat yourself much. Respond primarily in CASUAL COLLOQUIAL Devanagari Hindi (देवनागरी हिंदी) and "
                "use English only for some necessary english words and phrases in your responses. Keep history of the conversation in mind when answering and try not to repeat the same thing, "
                "if repetition is necessary try to change some words to keep conversation natural. You will be provided with some FAQ KNOWLEDGE BASE to answer user questions, "
                "use it wisely and use the info from context that is very relevant to the user's questions and not anything else. You may use acknowledgement like 'जी मैं बताती हूं..','ठीक है।' wherever relevant. "
                "Once you have answered the question stick to the conversation flow and your GOAL without diverting (for example if user ask about salary then tell the salary and get back to the goal and ask if they are interested in the job). "
                "CRITICAL: \n Use functions/tools ONLY AFTER the USER responds. If user is un-sure says stuff like 'pata nahi' try to convince them. \n Function calls are not going to have any arguments \n You will be provided a GOAL at stages of the call. NEVER ASSUME ANYTHING AND ADD ANY INFORMATION FROM YOUR SIDE."
                "You will be provided with a tool call to set the lead status. Use it wisely and use the info from context that is very relevant to the user's questions and not anything else."
                "Don't provide any information about the tool call in your response. Just use it to set the lead status."
                "Don't provide any other extra information after the user accepts/declines the job. And dont ask any other questions other than the interest in the job."
            )
        },
        {
            "role": "system",
            "content": (
                "Start by saying something like 'नमस्ते मैं वाहन जॉब्स से प्रिया बोल रही हूं। मेरे पास आपके लिए एक बढ़िया जॉब ऑफर है। क्या आप इंटरेस्टेड हैं?', "
                "if user asks a question, answer using information you have in context, then ask something like: 'kya आप इंटरेस्टेड हैं?' (Keep it varied and natural)\n"
                "- Use functions ONLY AFTER a clear yes/no. Just mention the 'jobs' and not 'delivery jobs' unless specifically asked about the job."
            )
        }
    ]
    context = OpenAILLMContext(messages, tools=tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        # DebugProcessor("PRE-STT"),
        stt,
        # DebugProcessor("POST-STT"),
        # FilterInterimFrames(),
        FilterAudioFrames(),
        # FilterTranscriptionFrames(),
        # DebugProcessor("Pre-classification"),
        classification_processor,
        # DebugProcessor("POST-classification"),
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,   # Twilio uses 8kHz
            audio_out_sample_rate=24000, # TTS outputs 16kHz, transport will resample
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    # await transport.start()
    run_task = asyncio.create_task(runner.run(task))
    await asyncio.sleep(2)  # Give the server a moment to start

    try:
        server_url = os.getenv("NGROK_URL")
        if not server_url:
            raise ValueError("NGROK_URL not set in .env file")

        call = twilio_client.calls.create(
            twiml = f'<Response><Connect><Stream url="{server_url}"></Stream></Connect></Response>',
            to=lead['phone_number'],
            from_=os.getenv("TWILIO_PHONE_NUMBER")
        )
        print(f"Call initiated with SID: {call.sid}")
        await run_task


    except Exception as e:
        print(f"Error: {e}")
    finally:
        final_status = classification_processor.final_lead_status
        final_summary = classification_processor.final_call_summary

        await runner.stop_when_done()
        await transport._stop_server()  # Stop the transport
        
        print(f"Updating lead with status: {final_status}")
        csv_manager.update_lead(
            lead_id=lead['lead_id'],
            new_status=final_status,
            summary=final_summary,
        )
        print("Lead updated.")


if __name__ == "__main__":
    asyncio.run(main())

