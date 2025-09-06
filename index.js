import { pipeline } from "@xenova/transformers";
import wavefile from "wavefile";
import fs from "fs";

const EMBED =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin";
const PHRASE =
  "Happy Friday! Today I want to share with you the fascinating TangoCode's case study, a company that has experienced impressive growth, and we are delighted to have been part of its journey.";

const synthesizer = await pipeline(
  "text-to-speech",
  "Xenova/speecht5_tts",
  { quantized: false }
);

const output = await synthesizer(PHRASE, {
  speaker_embeddings: EMBED,
});

const wav = new wavefile.WaveFile();
wav.fromScratch(
  1,
  output.sampling_rate,
  "32f",
  output.audio
);
fs.writeFileSync("out.wav", wav.toBuffer());

// https://github.com/xenova/transformers.js
// "type": "module",
// npm init -y
// npm i @xenova/transformers wavefile -E
// node index.js
