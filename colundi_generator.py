import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
import math
from enum import Enum
from typing import Callable, Dict, List


class WaveFormType(Enum):
    SINE = "sine"
    TRIANGLE = "triangle"
    SQUARE = "square"
    SAWTOOTH = "sawtooth"


class WaveformGenerator:
    def __init__(self, sample_rate: int, duration: int, amplitude: float = 0.5):
        self.sample_rate: int = sample_rate
        self.duration: int = duration
        self.amplitude: float = amplitude
        self.wave_functions: Dict[WaveFormType, Callable[[np.ndarray], np.ndarray]] = {
            WaveFormType.SINE: np.sin,
            WaveFormType.TRIANGLE: np.vectorize(self._triangle_func),
            WaveFormType.SQUARE: np.vectorize(self._square_func),
            WaveFormType.SAWTOOTH: np.vectorize(self._sawtooth_func),
        }

    def _generate_wave(
        self, wave_func: Callable[[np.ndarray], np.ndarray], frequency: float
    ) -> np.ndarray:
        num_periods: float = self.duration * frequency
        num_periods = math.ceil(num_periods)
        adjusted_duration: float = num_periods / frequency
        num_samples: int = int(self.sample_rate * adjusted_duration)
        t: np.ndarray = np.linspace(0, adjusted_duration, num_samples, endpoint=False)
        return self.amplitude * wave_func(2 * np.pi * frequency * t)

    def generate_files(
        self, frequencies: List[float], output_folder: str, waveform_type: WaveFormType
    ) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        wave_function: Callable[[np.ndarray], np.ndarray] = self.wave_functions[
            waveform_type
        ]
        for frequency in frequencies:
            generated_waveform: np.ndarray = self._generate_wave(
                wave_function, frequency
            )
            sf.write(
                str(output_folder / f"colundi_{frequency}.wav"),
                generated_waveform,
                self.sample_rate,
            )

    @staticmethod
    def _triangle_func(x: float) -> float:
        return (
            2 * np.abs(2 * ((x / (2 * np.pi)) - np.floor((x / (2 * np.pi)) + 0.5))) - 1
        )

    @staticmethod
    def _square_func(x: float) -> int:
        return 1 if (x % (2 * np.pi)) < np.pi else -1

    @staticmethod
    def _sawtooth_func(x: float) -> float:
        return 2 * (x / (2 * np.pi) - np.floor(0.5 + x / (2 * np.pi)))


def load_frequencies(file_path: str) -> List[float]:
    return [float(line.strip()) for line in Path(file_path).read_text().split("\n")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate waveforms.")
    parser.add_argument(
        "--sample_rate", type=int, default=96000, help="Sample rate of the waveform"
    )
    parser.add_argument(
        "--duration", type=int, default=1, help="Duration of the waveform in seconds"
    )
    parser.add_argument(
        "--amplitude", type=float, default=0.5, help="Amplitude of the waveform"
    )
    parser.add_argument(
        "--waveform",
        type=WaveFormType,
        choices=list(WaveFormType),
        default=WaveFormType.SINE,
    )
    args = parser.parse_args()
    frequencies: List[float] = load_frequencies("colundi_hertz.txt")
    output_folder = f"colundi_waveforms_{args.waveform.name.lower()}"

    print(f"Generating {args.waveform} waveforms for {len(frequencies)} frequencies")
    generator: WaveformGenerator = WaveformGenerator(
        sample_rate=args.sample_rate, duration=args.duration, amplitude=args.amplitude
    )
    generator.generate_files(frequencies, output_folder, args.waveform)
    print(f"Waveforms generated, outputs are in {output_folder}")


if __name__ == "__main__":
    main()
