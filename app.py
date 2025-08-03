#!/usr/bin/env python3
import base64
import io
import asyncio
import os
from functools import partial

from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from models.sdxl import SDXLModel
from models.aesthetic import AestheticScorer
from utils.logger import setup_logging

# Structured logging setup
logger = setup_logging()

# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "stabilityai/stable-diffusion-xl-base-1.0")

# Instantiate FastMCP with HTTP streaming enabled
mcp = FastMCP(
    name="StableMCP",
    stateless_http=True
)
# Initialize models
sdxl = SDXLModel(model_name=MODEL_NAME)
scorer = AestheticScorer()

# This is now a SYNCHRONOUS function
def progress_callback(loop: asyncio.AbstractEventLoop, ctx: Context, step: int, timestep: float, latents: dict):
    """
    Synchronous callback that schedules the async progress report on the main event loop.
    This function is called from a worker thread.
    """
    global total_steps
    # Create the coroutine we want to run
    coro = ctx.report_progress(step + 1, total_steps, f"step {step + 1}/{total_steps} done")
    # Submit the coroutine to the main event loop
    asyncio.run_coroutine_threadsafe(coro, loop)

    # The diffusers pipeline expects a dictionary or None as a return value.
    # We are not modifying the pipeline state, so we return an empty dictionary.
    return {}

@mcp.tool(annotations={"streamingHint": True})
async def generate_image(
    ctx: Context,
    prompt: str,
    steps: int = 10,
    retry_threshold: float = 10.0,
    max_retries: int = 3,
    timeout: float = 300.0,
) -> str:
    """
    Generate an image via StableDiffusionXL, score it, and retry up to ``max_retries``
    if the aesthetic score is below ``retry_threshold``. Streams step progress,
    then returns the best-scoring image as Base64‑encoded PNG.

    Parameters
    ----------
    ctx : Context
        MCP execution context used for progress streaming.
    prompt : str
        Text prompt for the diffusion model.
    steps : int, optional
        Number of inference steps for each attempt.
    retry_threshold : float, optional
        Minimum acceptable aesthetic score before stopping retries.
    max_retries : int, optional
        Maximum number of generation attempts.
    timeout : float, optional
        Maximum seconds to wait for a single generation attempt before aborting.
    """
    best_image = None
    best_score = -1.0

    # This global is used by the callback to know the total number of steps.
    global total_steps
    total_steps = steps

    # Get the current event loop to pass to the thread-safe callback
    loop = asyncio.get_running_loop()

    for attempt in range(1, max_retries + 1):
        # Create a partial function to pass the LOOP and the CONTEXT to the callback
        callback_with_context = partial(progress_callback, loop, ctx)

        # Run the synchronous SDXL generation in a separate thread with timeout
        try:
            output_images = await asyncio.wait_for(
                loop.run_in_executor(
                    None,  # Use the default thread pool executor
                    partial(
                        sdxl.generate,
                        prompt,
                        num_images=1,
                        num_inference_steps=steps,
                        callback=callback_with_context,  # diffusers accepts `callback`
                        callback_steps=1,  # call the callback every step
                    ),
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            logger.error(f"Image generation timed out after {timeout} seconds")
            raise TimeoutError(
                f"Image generation timed out after {timeout} seconds"
            ) from e
        
        buf = io.BytesIO()
        output_images[0].save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Score the image
        score = scorer.score(img_bytes)
        logger.info(f"Attempt {attempt} score: {score}")

        # Keep track of the best result
        if score > best_score:
            best_score = score
            best_image = img_bytes

        # Stop early if we meet the threshold
        if score >= retry_threshold:
            break

    # Return best image as Base64
    return base64.b64encode(best_image).decode("utf-8")

@mcp.tool()
async def aesthetic_score(image_b64: str) -> float:
    """
    Decode a Base64‑encoded PNG, run the aesthetic scorer, and return a float score.
    """
    img_bytes = base64.b64decode(image_b64)
    return scorer.score(img_bytes)

if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
    )
