import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import librosa
import soundfile as sf
import pygame
from transformers import Wav2Vec2Model
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import threading
import torch

from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier



# FUNZIONI SUPPORTO MODELLO
class spoofing_pipeline(nn.Module):
        def __init__(self,wav2vec2, aasist, layer_index = 4):

            super().__init__()
            self.layer_index = layer_index
            self.wav2vec2 = wav2vec2
            self.aasist = aasist

            for p in self.wav2vec2.parameters():
                p.requires_grad = False

            # congela AASIST
            for p in self.aasist.parameters():
                p.requires_grad = False
        
        def zero_mean_unit_var_norm_torch(self, x, attention_mask=None, padding_value=0.0, eps=1e-7):
            """
            x: (B, T) float
            attention_mask: (B, T) {0,1} oppure None
            """
            if attention_mask is None:
                mean = x.mean(dim=-1, keepdim=True)
                var  = x.var(dim=-1, keepdim=True, unbiased=False)  # ddof=0 come numpy.var()
                return (x - mean) / torch.sqrt(var + eps)

            mask = attention_mask.to(dtype=x.dtype)
            lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

            mean = (x * mask).sum(dim=-1, keepdim=True) / lengths
            var  = ((x - mean) ** 2 * mask).sum(dim=-1, keepdim=True) / lengths

            x_norm = (x - mean) / torch.sqrt(var + eps)
            x_norm = x_norm * mask + padding_value * (1.0 - mask)
            return x_norm

        def reshape_features_for_model(self, features, model_name):
            """Adatta le features al modello specifico"""
            # features arriva come [batch, seq_len, hidden_dim] = [64, 49, 1024]
            if model_name == 'W2VAASIST':
                # W2VAASIST vuole [batch, hidden_dim, seq_len] NON 4D!
                return features.transpose(1, 2)  # [64, 1024, 49]
            else:
                return features

        def forward(self, waveform):
            # Ottengo waveform paddata e in formato mono 16kHz
            #waveform, sr = load_16k_mono(waveform)

            if waveform.dim() == 3:
                waveform = waveform.squeeze(1)
            
            waveform_norm = self.zero_mean_unit_var_norm_torch(waveform)

            #mid_features = self.wav2vec2(waveform_norm).hidden_states[-1]
            mid_features = self.wav2vec2(waveform_norm).last_hidden_state
            mid_features = self.reshape_features_for_model(mid_features, 'W2VAASIST')
            _, logits = self.aasist(mid_features)
            return logits

class AudioDeepfakeGUI:
    """
    Classe principale dell'interfaccia grafica.

    Questa classe gestisce:
    - caricamento del file audio;
    - selezione del rumore ambientale o dell'attacco adversariale;
    - scelta del livello di rumore tramite SNR in dB;
    - generazione dell'audio modificato;
    - riproduzione del rumore e dell'audio modificato;
    - invio dell'audio modificato al detector;
    - visualizzazione del risultato finale: spoof oppure bonafide.
    """

    def __init__(self, root):
        """
        Costruttore della GUI.

        Inizializza la finestra principale, le variabili di stato,
        il mixer audio per la riproduzione e crea tutti i componenti grafici.
        """

        self.root = root
        self.root.title("Audio Deepfake Detector")
        self.configure_window_size(desired_width=850, desired_height=680)

        # Inizializza pygame per poter riprodurre file audio.
        pygame.mixer.init()

        # Path del file audio originale caricato dall'utente.
        self.audio_path = None

        # Path del file audio modificato con rumore o attacco adversariale.
        self.modified_audio_path = None

        # Path del rumore ambientale selezionato.
        self.selected_noise_path = None

        # Variabile associata alla scelta nella combobox.
        self.selected_noise_or_attack = tk.StringVar(value="Nessuna selezione")

        # Variabile associata al risultato del detector.
        self.detector_result = tk.StringVar(value="Risultato non disponibile")

        #Variabili per il caricamento del modello
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.model_loading_error = None
        

        # Valore SNR scelto dall'utente.
        # SNR più alto significa rumore meno invasivo.
        # SNR più basso significa rumore più forte.
        self.snr_db = tk.DoubleVar(value=10.0)

        # Dizionario che associa il nome mostrato nella GUI al file audio del rumore.
        self.noise_files = {
            "Vento": "rumori/vento.wav",
            "Rumore di folla": "rumori/folla.wav",
            "Rumore di bambini": "rumori/bambini.wav",
            "Pioggia": "rumori/pioggia.wav"
        }

        # Cartella in cui vengono salvati gli audio modificati.
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Creazione degli elementi grafici.
        self.create_widgets()
        #Thread per caricamento parallelo
        self.start_model_loading()

    def configure_window_size(self, desired_width=850, desired_height=680):
        """
        Configura la dimensione della finestra in modo che non superi
        mai la dimensione dello schermo.

        Se lo schermo è più grande della dimensione desiderata, usa:
            desired_width x desired_height

        Se lo schermo è più piccolo, riduce automaticamente la finestra
        lasciando un piccolo margine.
        """

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        margin = 80

        max_width = screen_width - margin
        max_height = screen_height - margin

        window_width = min(desired_width, max_width)
        window_height = min(desired_height, max_height)

        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        self.root.geometry(
            f"{window_width}x{window_height}+{x_position}+{y_position}"
        )

        self.root.minsize(700, 500)

        self.root.resizable(True, True)

    def create_widgets(self):
        """
        Crea la struttura principale dell'interfaccia grafica.

        Questa funzione costruisce:
        - il titolo della finestra;
        - la sezione di caricamento audio;
        - la sezione del detector;
        - la sezione di output;
        - la sezione per scegliere il livello di rumore;
        - la sezione per ascoltare gli audio.
        """

        title_label = tk.Label(
            self.root,
            text="Interfaccia per Audio Deepfake Detection",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=15)

        main_frame = tk.Frame(self.root)
        main_frame.pack(pady=5, padx=20, fill="both", expand=True)

        left_frame = tk.LabelFrame(
            main_frame,
            text="Input audio",
            font=("Arial", 11, "bold"),
            padx=15,
            pady=15
        )
        left_frame.grid(row=0, column=0, padx=15, pady=10, sticky="nsew")

        center_frame = tk.LabelFrame(
            main_frame,
            text="Detector",
            font=("Arial", 11, "bold"),
            padx=15,
            pady=15
        )
        center_frame.grid(row=0, column=1, padx=15, pady=10, sticky="nsew")

        right_frame = tk.LabelFrame(
            main_frame,
            text="Output",
            font=("Arial", 11, "bold"),
            padx=15,
            pady=15
        )
        right_frame.grid(row=0, column=2, padx=15, pady=10, sticky="nsew")

        load_button = tk.Button(
            left_frame,
            text="Carica file audio",
            width=22,
            command=self.load_audio
        )
        load_button.pack(pady=10)

        self.audio_label = tk.Label(
            left_frame,
            text="Nessun file caricato",
            wraplength=200,
            fg="gray"
        )
        self.audio_label.pack(pady=10)

        selection_label = tk.Label(
            left_frame,
            text="Scegli rumore o attacco:"
        )
        selection_label.pack(pady=(20, 5))

        self.combo = ttk.Combobox(
            left_frame,
            textvariable=self.selected_noise_or_attack,
            values=[
                "Vento",
                "Rumore di folla",
                "Rumore di bambini",
                "Pioggia",
                "Attacco adversariale PGD"
            ],
            state="readonly",
            width=25
        )
        self.combo.pack(pady=5)

        # Quando l'utente seleziona una voce nella combobox,
        # viene chiamata la funzione on_operation_selected.
        self.combo.bind("<<ComboboxSelected>>", self.on_operation_selected)

        detector_label = tk.Label(
            center_frame,
            text="Audio modificato → Detector → Classificazione",
            font=("Arial", 12),
            wraplength=200
        )
        detector_label.pack(pady=30)

        # Label di caricamento pesi modello
        self.model_status_label = tk.Label(
            center_frame,
            text="Caricamento modello in corso...",
            fg="orange",
            wraplength=200
        )
        self.model_status_label.pack(pady=5)

        process_button = tk.Button(
            center_frame,
            text="Modifica audio",
            width=22,
            command=self.process_audio
        )
        process_button.pack(pady=10)

        run_button = tk.Button(
            center_frame,
            text="Esegui detector",
            width=22,
            command=self.run_detector
        )
        run_button.pack(pady=10)

        result_title = tk.Label(
            right_frame,
            text="Risultato detector:",
            font=("Arial", 12, "bold")
        )
        result_title.pack(pady=10)

        result_label = tk.Label(
            right_frame,
            textvariable=self.detector_result,
            font=("Arial", 16, "bold"),
            fg="blue",
            wraplength=180
        )
        result_label.pack(pady=30)

        # Crea la sezione SNR, inizialmente nascosta.
        self.create_snr_section()

        # Crea la sezione per ascoltare rumore e audio modificato.
        self.create_audio_player_section()

    def create_snr_section(self):
        """
        Crea una sezione compatta per scegliere il livello di rumore ambientale.

        Questa sezione viene mostrata solo quando l'utente seleziona
        uno dei rumori ambientali.

        Rispetto alla versione precedente, occupa meno spazio verticale perché
        usa una disposizione orizzontale con grid().
        """

        self.snr_frame = tk.LabelFrame(
            self.root,
            text="Livello rumore ambientale",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=8
        )

        snr_text_label = tk.Label(
            self.snr_frame,
            text="SNR:",
            font=("Arial", 10)
        )
        snr_text_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.snr_value_label = tk.Label(
            self.snr_frame,
            text=f"{self.snr_db.get():.1f} dB",
            font=("Arial", 10, "bold"),
            width=8
        )
        self.snr_value_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        snr_slider = tk.Scale(
            self.snr_frame,
            from_=0,
            to=45,
            resolution=0.5,
            orient="horizontal",
            length=350,
            variable=self.snr_db,
            command=self.update_snr_label,
            showvalue=False
        )
        snr_slider.grid(row=0, column=2, padx=10, pady=5, sticky="ew")

        snr_hint = tk.Label(
            self.snr_frame,
            text="0 dB = forte, 10 dB = medio, 20 dB = leggero",
            fg="gray",
            font=("Arial", 9)
        )
        snr_hint.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        self.snr_frame.grid_columnconfigure(2, weight=1)

        # All'inizio la sezione SNR è nascosta.
        self.snr_frame.pack_forget()

    def create_audio_player_section(self):
        """
        Crea la sezione inferiore dedicata all'ascolto.

        Questa sezione permette di:
        - ascoltare l'audio originale selezionato da file;
        - ascoltare il rumore ambientale selezionato;
        - ascoltare l'audio modificato;
        - interrompere la riproduzione audio.
        """

        bottom_frame = tk.LabelFrame(
            self.root,
            text="Ascolto audio",
            font=("Arial", 11, "bold"),
            padx=15,
            pady=15
        )
        bottom_frame.pack(pady=10, padx=20, fill="x")

        play_original_button = tk.Button(
            bottom_frame,
            text="Ascolta audio originale",
            width=25,
            command=self.play_original_audio
        )
        play_original_button.grid(row=0, column=0, padx=10, pady=5)

        play_noise_button = tk.Button(
            bottom_frame,
            text="Ascolta rumore selezionato",
            width=25,
            command=self.play_selected_noise
        )
        play_noise_button.grid(row=0, column=1, padx=10, pady=5)

        play_modified_button = tk.Button(
            bottom_frame,
            text="Ascolta audio modificato",
            width=25,
            command=self.play_modified_audio
        )
        play_modified_button.grid(row=0, column=2, padx=10, pady=5)

        stop_button = tk.Button(
            bottom_frame,
            text="Stop",
            width=12,
            command=self.stop_audio
        )
        stop_button.grid(row=0, column=3, padx=10, pady=5)

        self.status_label = tk.Label(
            bottom_frame,
            text="Nessun audio in riproduzione",
            fg="gray"
        )
        self.status_label.grid(row=1, column=0, columnspan=4, pady=10)


    def play_original_audio(self):
        """
        Riproduce l'audio originale caricato dall'utente.

        Questa funzione controlla che l'utente abbia prima selezionato
        un file audio tramite il bottone 'Carica file audio'.

        Se il file esiste, viene riprodotto usando pygame.
        """

        if self.audio_path is None:
            messagebox.showwarning(
                "Attenzione",
                "Devi prima caricare un file audio."
            )
            return

        if not os.path.exists(self.audio_path):
            messagebox.showerror(
                "Errore",
                "Il file audio originale non esiste più nel percorso selezionato."
            )
            return

        self.play_audio(self.audio_path)

        self.status_label.config(
            text="In riproduzione: audio originale",
            fg="black"
        )

    def update_snr_label(self, value):
        """
        Aggiorna il testo che mostra il valore SNR selezionato.
        """

        self.snr_value_label.config(
            text=f"{float(value):.1f} dB"
        )

        self.modified_audio_path = None
        self.detector_result.set("Risultato non disponibile")

    def load_audio(self):
        """
        Permette all'utente di caricare un file audio dal filesystem.

        Quando l'utente seleziona un file:
        - il path viene salvato in self.audio_path;
        - l'etichetta grafica viene aggiornata;
        - l'audio modificato precedente viene eliminato logicamente;
        - il risultato del detector viene resettato.
        """

        file_path = filedialog.askopenfilename(
            title="Seleziona un file audio",
            filetypes=[
                ("File audio", "*.wav *.mp3 *.flac *.ogg"),
                ("Tutti i file", "*.*")
            ]
        )

        if file_path:
            self.audio_path = file_path
            self.modified_audio_path = None
            self.audio_label.config(text=file_path, fg="black")
            self.detector_result.set("Risultato non disponibile")

    def on_operation_selected(self, event=None):
        """
        Gestisce la selezione del rumore ambientale o dell'attacco PGD.

        Questa funzione viene chiamata quando l'utente cambia scelta
        nella combobox.

        Se viene selezionato un rumore ambientale, la GUI mostra la sezione
        per scegliere il valore SNR in dB.

        Se viene selezionato l'attacco adversariale PGD, la sezione SNR
        viene nascosta, perché PGD non usa un rumore ambientale esterno.
        """

        operation = self.selected_noise_or_attack.get()

        self.modified_audio_path = None
        self.detector_result.set("Risultato non disponibile")

        if operation in self.noise_files:
            self.selected_noise_path = self.noise_files[operation]

            # Mostra la sezione SNR prima della sezione di ascolto.
            self.snr_frame.pack(
                pady=10,
                padx=20,
                fill="x",
                before=self.root.winfo_children()[-1]
            )

        else:
            self.selected_noise_path = None

            # Nasconde la sezione SNR se non è stato selezionato un rumore ambientale.
            self.snr_frame.pack_forget()

    def process_audio(self):
        """
        Avvia la modifica dell'audio caricato.

        Questa funzione controlla che:
        - sia stato caricato un file audio;
        - sia stata scelta un'operazione tra rumore ambientale e PGD.

        Successivamente richiama apply_noise_or_attack, che produce
        l'audio modificato e salva il path risultante in self.modified_audio_path.
        """

        if self.audio_path is None:
            messagebox.showwarning(
                "Attenzione",
                "Devi prima caricare un file audio."
            )
            return

        operation = self.selected_noise_or_attack.get()

        if operation == "Nessuna selezione":
            messagebox.showwarning(
                "Attenzione",
                "Devi scegliere un rumore oppure l'attacco PGD."
            )
            return

        self.modified_audio_path = self.apply_noise_or_attack(
            audio_path=self.audio_path,
            operation=operation
        )

        if self.modified_audio_path is not None:
            messagebox.showinfo(
                "Audio modificato",
                f"Audio modificato salvato in:\n{self.modified_audio_path}"
            )

    def apply_noise_or_attack(self, audio_path, operation):
        """
        Applica all'audio originale l'operazione scelta dall'utente.

        Se l'utente ha scelto un rumore ambientale, questa funzione:
        - recupera il file del rumore corrispondente;
        - legge il valore SNR scelto dall'utente;
        - richiama add_noise_to_audio per generare l'audio rumoroso.

        Se l'utente ha scelto PGD, questa funzione richiama il placeholder
        generate_pgd_attack_placeholder, che potrai sostituire con il tuo
        vero attacco adversariale.
        """

        if operation in self.noise_files:
            noise_path = self.noise_files[operation]

            if not os.path.exists(noise_path):
                messagebox.showerror(
                    "Errore",
                    f"File del rumore non trovato:\n{noise_path}"
                )
                return None

            selected_snr_db = self.snr_db.get()

            return self.add_noise_to_audio(
                audio_path=audio_path,
                noise_path=noise_path,
                output_path=os.path.join(self.output_dir, "audio_modificato.wav"),
                snr_db=selected_snr_db
            )

        elif operation == "Attacco adversariale PGD":
            return self.generate_pgd_attack_placeholder(
                audio_path=audio_path,
                output_path=os.path.join(self.output_dir, "audio_pgd.wav")
            )

        return None


    def torchaudio_load(filepath):
        wave, sr = librosa.load(filepath, sr=16000)
        waveform = torch.Tensor(np.expand_dims(wave, axis=0))
        return [waveform, sr]

    def fit_noise_audio_to_x(self,
    x: torch.Tensor,
    noise_audio: torch.Tensor,
    crop_mode: str = "repeat"
):
        """
        Adatta un audio-rumore alla shape di x.

        Parametri
        ---------
        x : torch.Tensor
            shape (B, T) oppure (B, 1, T)

        noise_audio : torch.Tensor
            può avere shape:
            - (Tn,)
            - (1, Tn)
            - (B, Tn)
            - (B, 1, Tn)

        crop_mode : str
            - "repeat": se il rumore è più corto, lo ripete fino a coprire T
            - "center_crop": se il rumore è più lungo, prende il centro
            - "random_crop": se il rumore è più lungo, prende una finestra casuale

        Ritorna
        -------
        noise_fitted : torch.Tensor
            stessa shape di x
        """
        if x.ndim not in (2, 3):
            raise ValueError(f"x must have shape (B,T) or (B,1,T), got {x.shape}")

        B = x.shape[0]
        T = x.shape[-1]

        # Porta noise_audio a shape (Bn, Tn)
        if noise_audio.ndim == 1:
            noise_flat = noise_audio.unsqueeze(0)         # (1, Tn)
        elif noise_audio.ndim == 2:
            noise_flat = noise_audio                      # (Bn, Tn)
        elif noise_audio.ndim == 3:
            noise_flat = noise_audio.reshape(noise_audio.shape[0], -1)
        else:
            raise ValueError(f"Unsupported noise_audio shape: {noise_audio.shape}")

        Bn, Tn = noise_flat.shape

        # Se c'è un solo noise audio, lo condivido per tutto il batch
        if Bn == 1:
            noise_flat = noise_flat.repeat(B, 1)
        elif Bn != B:
            raise ValueError(
                f"noise_audio batch size must be 1 or equal to x batch size. "
                f"Got noise batch={Bn}, x batch={B}"
            )

        fitted = []
        for i in range(B):
            n = noise_flat[i]

            if Tn == T:
                n_fit = n

            elif Tn < T:
                # ripeti fino a raggiungere T
                reps = math.ceil(T / Tn)
                n_fit = n.repeat(reps)[:T]

            else:  # Tn > T
                if crop_mode == "center_crop":
                    start = (Tn - T) // 2
                elif crop_mode == "random_crop":
                    start = torch.randint(0, Tn - T + 1, (1,), device=n.device).item()
                else:
                    # default: prendi l'inizio
                    start = 0
                n_fit = n[start:start + T]

            fitted.append(n_fit)

        noise_fitted = torch.stack(fitted, dim=0)  # (B, T)

        if x.ndim == 3:
            noise_fitted = noise_fitted.unsqueeze(1)  # (B, 1, T)

        return noise_fitted

    def compute_snr_db(self, x_adv: torch.Tensor, x: torch.Tensor, eps: float = 1e-12):
        """
        x, x_adv: shape (B, T) oppure (B, 1, T)
        ritorna: SNR per campione, shape (B,)
        """
        if x_adv.shape != x.shape:
            raise ValueError(f"Shapes of x_adv {x_adv.shape} and x {x.shape} must match")

        delta = x_adv - x

        x_flat = x.reshape(x.shape[0], -1)
        delta_flat = delta.reshape(delta.shape[0], -1)

        p_signal = torch.sum(x_flat ** 2, dim=1)
        p_noise = torch.sum(delta_flat ** 2, dim=1) + eps

        return 10.0 * torch.log10(p_signal / p_noise)
    
    def make_audio_with_target_snr_from_noise_audio(self,
            x: torch.Tensor,
            noise_audio: torch.Tensor,
            snr_target_db,
            crop_mode: str = "repeat",
            eps: float = 1e-12
        ):
        """
        Costruisce x_adv = x + delta usando come perturbazione un audio reale (es. 1 secondo),
        scalato in modo da ottenere un SNR target scelto a priori.

        Parametri
        ---------
        x : torch.Tensor
            shape (B, T) oppure (B, 1, T)

        noise_audio : torch.Tensor
            audio-rumore base, es. 1 secondo
            shape supportate:
            - (Tn,)
            - (1, Tn)
            - (B, Tn)
            - (B, 1, Tn)

        snr_target_db : float | list | torch.Tensor
            SNR desiderato in dB
            - float: stesso SNR per tutto il batch
            - list/tensor di lunghezza B: SNR diverso per campione

        crop_mode : str
            strategia per adattare il noise audio alla lunghezza di x

        Ritorna
        -------
        x_adv : torch.Tensor
        delta : torch.Tensor
        actual_snr : torch.Tensor
        """
        if x.ndim not in (2, 3):
            print("ooooo sample")
            x = x.unsqueeze(0)
        

        B = x.shape[0]

        # 1) adatta il noise audio alla lunghezza di x
        noise = self.fit_noise_audio_to_x(x, noise_audio, crop_mode=crop_mode)

        # 2) flatten per calcolo potenze
        x_flat = x.reshape(B, -1) # se (b,1,16k) trasforma in (B,16k)
        noise_flat = noise.reshape(B, -1)

        p_signal = torch.sum(x_flat ** 2, dim=1, keepdim=True)               # (B,1)
        p_noise_raw = torch.sum(noise_flat ** 2, dim=1, keepdim=True) + eps  # (B,1)

        # 3) gestisci SNR target
        if isinstance(snr_target_db, (float, int)):
            snr_target_db = torch.full(
                (B, 1), float(snr_target_db), device=x.device, dtype=x.dtype
            )
        else:
            snr_target_db = torch.as_tensor(
                snr_target_db, device=x.device, dtype=x.dtype
            ).reshape(B, 1)

        # 4) conversione dB -> lineare
        target_linear = 10.0 ** (snr_target_db / 10.0)

        # 5) scala il noise audio
        alpha = torch.sqrt(p_signal / (p_noise_raw * target_linear))         # (B,1)

        delta_flat = alpha * noise_flat
        delta = delta_flat.reshape_as(x)

        # 6) audio finale
        x_adv = x + delta

        # 7) verifica
        actual_snr = self.compute_snr_db(x_adv, x, eps=eps)

        return x_adv, delta, actual_snr

    def add_noise_to_audio(self, audio_path, noise_path, output_path, snr_db=10):
        """
        Aggiunge un rumore ambientale all'audio originale controllando l'SNR.

        Questa funzione:
        - carica il segnale audio originale;
        - carica il rumore ambientale;
        - adatta la durata del rumore alla durata dell'audio originale;
        - calcola la potenza media dell'audio e del rumore;
        - scala il rumore per ottenere l'SNR desiderato;
        - somma audio e rumore;
        - normalizza il risultato per evitare clipping;
        - salva il file audio modificato.

        Il parametro snr_db controlla quanto rumore viene aggiunto.

        Esempi:
        - snr_db = 20 significa rumore leggero;
        - snr_db = 10 significa rumore medio;
        - snr_db = 5 significa rumore forte;
        - snr_db = 0 significa rumore molto forte.
        """

        #audio, sr_audio = librosa.load(audio_path, sr=None, mono=True)
        #noise, sr_noise = librosa.load(noise_path, sr=sr_audio, mono=True)

        audio, sr = self.load_16k_mono(audio_path, target_sr=16000)
        audio = self.pad_dataset(audio, target_seconds=1.0, sample_rate=sr)

        noise, sr = self.load_16k_mono(noise_path, target_sr=16000)
        noise = self.pad_dataset(noise, target_seconds=1.0, sample_rate=sr)

        noisy_audio, delta, actual_snr = self.make_audio_with_target_snr_from_noise_audio(x=audio, noise_audio=noise, snr_target_db=snr_db, crop_mode="repeat")

        # Assicura che la cartella di output esista.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Ferma e scarica eventuale audio in riproduzione.
        # Questo è importante su Windows, perché pygame può bloccare il file.
        pygame.mixer.music.stop()

        try:
            pygame.mixer.music.unload()
        except Exception:
            pass

        # Salvataggio dell'audio finale modificato.
        sr_audio = 16000
        noisy_audio_np = noisy_audio.squeeze(0).detach().cpu().numpy()
        sf.write(output_path, noisy_audio_np, sr_audio)

        return output_path

    def match_length(self, signal, target_length):
        """
        Adatta la lunghezza di un segnale alla lunghezza desiderata.

        Se il rumore è più lungo dell'audio, viene tagliato.

        Se il rumore è più corto dell'audio, viene ripetuto più volte
        fino a raggiungere la lunghezza dell'audio originale.

        Questa funzione è utile perché audio e rumore devono avere
        la stessa lunghezza per poter essere sommati campione per campione.
        """

        if len(signal) >= target_length:
            return signal[:target_length]

        repetitions = int(np.ceil(target_length / len(signal)))
        repeated_signal = np.tile(signal, repetitions)

        return repeated_signal[:target_length]

    def normalize_audio(self, audio):
        """
        Normalizza un segnale audio nell'intervallo sicuro [-0.95, 0.95].

        Questa funzione evita che il segnale superi il range ammesso
        e quindi riduce il rischio di clipping quando l'audio viene salvato.

        Se il segnale è completamente nullo, viene restituito senza modifiche.
        """

        max_value = np.max(np.abs(audio))

        if max_value == 0:
            return audio

        return audio / max_value * 0.95

    
    def generate_pgd_attack_placeholder(self, audio_path, output_path):
        """
        Genera un attacco adversariale PGD usando ART.

        La funzione:
        - carica l'audio a 16 kHz mono;
        - lo porta a 1 secondo;
        - crea un batch di dimensione 1;
        - converte l'audio in NumPy per ART;
        - genera l'audio adversariale;
        - salva il risultato su file.
        """

        if not self.model_loaded or self.model is None:
            messagebox.showwarning(
                "Modello non pronto",
                "Il modello non è ancora stato caricato."
            )
            return None

        audio, sr = self.load_16k_mono(audio_path, target_sr=16000)

        audio = self.pad_dataset(
            audio,
            target_seconds=1.0,
            sample_rate=sr
        )

        # Dopo pad_dataset:
        # audio.shape = torch.Size([16000])
        print("Audio dopo pad_dataset:", type(audio), audio.shape)

        # Aggiungo la dimensione batch:
        # da (16000,) a (1, 16000)
        audio = audio.unsqueeze(0)

        print("Audio con batch:", type(audio), audio.shape)

        # ART vuole NumPy in input.
        # Non spostare manualmente audio su cuda.
        audio_np = audio.detach().cpu().numpy().astype(np.float32)

        print("Audio NumPy per ART:", type(audio_np), audio_np.shape)

        classifier_art = PyTorchClassifier(
            model=self.model,
            loss=nn.CrossEntropyLoss(),
            input_shape=(16000,),
            nb_classes=2,
            device_type="gpu" if self.device.type == "cuda" else "cpu"
        )

        epsilon = 0.1
        num_iter = 7
        alpha = epsilon / num_iter

        attack = ProjectedGradientDescentPyTorch(
            estimator=classifier_art,
            eps=epsilon,
            eps_step=alpha,
            targeted=False,
            max_iter=num_iter,
            batch_size=1
        )

        # ART ritorna NumPy, non torch.Tensor.
        adversarial_audio_np = attack.generate(x=audio_np)

        print("Audio adversariale NumPy:", type(adversarial_audio_np), adversarial_audio_np.shape)

        # Da (1, 16000) a (16000,)
        adversarial_audio_np = adversarial_audio_np.squeeze(0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        pygame.mixer.music.stop()

        try:
            pygame.mixer.music.unload()
        except Exception:
            pass

        sf.write(output_path, adversarial_audio_np, 16000)

        return output_path
    

    # Funzione callback caricamento modello
    
    def on_model_loaded(self):
        """
        Aggiorna la GUI quando il modello è stato caricato correttamente.

        Questa funzione viene chiamata tramite root.after, quindi viene eseguita
        nel thread principale di tkinter.
        """

        self.model_status_label.config(
            text="Modello caricato correttamente",
            fg="green"
        )

    # Funzione callback fallimento caricamento modello
    def on_model_loading_failed(self):
        """
        Aggiorna la GUI quando il caricamento del modello fallisce.

        Mostra nella label un messaggio di errore sintetico.
        """

        self.model_status_label.config(
            text="Errore nel caricamento del modello",
            fg="red"
        )

        messagebox.showerror(
            "Errore caricamento modello",
            f"Non è stato possibile caricare il modello:\n{self.model_loading_error}"
        )


    
    

    # Funzione caricamento modello
    def load_model_in_background(self):
        """
        Carica il modello e i suoi pesi in background.

        Questa funzione viene eseguita in un thread separato.
        Qui devi creare il modello, caricare i pesi e spostarlo sul device corretto.

        Non bisogna aggiornare direttamente la GUI da questa funzione.
        Per aggiornare tkinter in sicurezza si usa self.root.after(...).
        """
        model_path = "../Models/cotrain_W2VAASIST_csam_v3/anti-spoofing_feat_model.pt"
        try:
            model_type = "light"
            MODEL = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-xls-r-300m",revision="refs/pr/15", use_safetensors=True,
            )
            MODEL.config.output_hidden_states = True
            print("Modello Wav2Vec2 caricato.")

            # se scelgo la versione light, prendo solo i primi k layer di wav2vec2 e disabilito lo stable layer norm, che è inutile se ho pochi layer e rallenta molto l'inferenza  
            if model_type == "light":
                print("MODEL LIGHT")
                k = 5
                # tengo solo i primi k layer transformer
                MODEL.encoder.layers = nn.ModuleList(list(MODEL.encoder.layers)[:k])
                # aggiorno anche la config per coerenza
                MODEL.config.num_hidden_layers = k
                if MODEL.config.do_stable_layer_norm:
                    MODEL.encoder.layer_norm = nn.Identity()
            #print("MODEL FULL")
            MODEL.to(self.device)
            MODEL.eval()    
        
            # Carico l'intero modello compreso di pesi .pt
            ADD_MODEL = torch.load(model_path, map_location=self.device, weights_only=False)
            ADD_MODEL.eval()
            print("Modello AASIST caricato.")

            self.model = spoofing_pipeline(MODEL, ADD_MODEL).to(self.device)
            self.model.eval()
            print("PIPELINE AASIST caricato.")
            self.model_loaded = True
            self.model_loading_error = None

            # Aggiornamento sicuro della GUI dal thread principale.
            self.root.after(0, self.on_model_loaded)

        except Exception as e:
            self.model = None
            self.model_loaded = False
            self.model_loading_error = str(e)

            # Aggiornamento sicuro della GUI in caso di errore.
            self.root.after(0, self.on_model_loading_failed) 



    # Funzione caricamento thread funzione caricamento pesi modello
    def start_model_loading(self):
        """
        Avvia il caricamento del modello in un thread separato.

        Questa funzione viene chiamata quando la GUI viene aperta.
        Il suo ruolo è creare un thread secondario che carica i pesi del modello
        senza bloccare il thread principale di tkinter.
        """

        loading_thread = threading.Thread(
            target=self.load_model_in_background,
            daemon=True
        )

        loading_thread.start()


    def run_detector(self):
        """
        Esegue il detector sull'audio modificato.

        Questa funzione controlla che:
        - sia stato caricato un audio originale;
        - sia stato prodotto un audio modificato.

        Poi passa l'audio modificato alla funzione detector e mostra
        nella GUI il risultato ottenuto.
        """

        if self.audio_path is None:
            messagebox.showwarning(
                "Attenzione",
                "Devi prima caricare un file audio."
            )
            return

        if self.modified_audio_path is None:
            messagebox.showwarning(
                "Attenzione",
                "Devi prima modificare l'audio."
            )
            return

        result = self.detector(self.modified_audio_path)

        self.detector_result.set(result.upper())

    def load_16k_mono(self, path, target_sr=16000):
        wav, sr = torchaudio.load(path)  # (C, T)
        # Portiamo in mono e a 16kHz se necessario
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)

        wav = wav.float()     # (1,T)
        return wav, target_sr
    
    def pad_dataset(self, wav, target_seconds=1.0, sample_rate=16000, crop = "center_crop"):

        waveform = wav.squeeze(0) if wav.dim() > 1 else wav
        waveform_len = waveform.shape[0]
        target_length = int(target_seconds * sample_rate)

        if waveform_len >= target_length:
            if crop == "center_crop":
                start_idx = (waveform_len - target_length) // 2
                return waveform[start_idx : start_idx + target_length]
            
            return waveform[:target_length]
        
        num_repeats = (target_length + waveform_len - 1) // waveform_len
        padded_waveform = waveform.repeat(num_repeats)[:target_length]
        return padded_waveform

    def detector(self, processed_audio):
            """
            Esegue l'inferenza del modello sull'audio modificato.

            Questa funzione viene chiamata quando l'utente preme 'Esegui detector'.
            Prima controlla che il modello sia stato caricato correttamente.
            """

            if not self.model_loaded or self.model is None:
                messagebox.showwarning(
                    "Modello non pronto",
                    "Il modello non è ancora stato caricato. Attendi il completamento del caricamento."
                )
                return "non disponibile"

            
            #sf.write(output_path, adversarial_audio, sr)
            waveform, sr = self.load_16k_mono(processed_audio, target_sr=16000)
            waveform = waveform.to(self.device)
            threshold = 0.7
            
            


            #waveform = torch.tensor(waveform, dtype=torch.float32)
            #waveform = waveform.unsqueeze(0)
            #waveform = waveform.to(self.device)

            with torch.no_grad():
                output = self.model(waveform)

                # Esempio per classificatore binario con un solo logit.
                probs = torch.softmax(output, dim=1)
                score_by_fake = probs[:, 1].cpu().numpy()

            
            print(score_by_fake)
            if score_by_fake >= threshold:
                return "spoof"
            else:
                return "bonafide"

    def play_selected_noise(self):
        """
        Riproduce il rumore ambientale selezionato dall'utente.

        Questa funzione funziona solo se è stato selezionato uno dei quattro
        rumori ambientali.

        Se invece è stato selezionato PGD, viene mostrato un messaggio,
        perché PGD non è un rumore esterno ascoltabile separatamente.
        """

        operation = self.selected_noise_or_attack.get()

        if operation == "Attacco adversariale PGD":
            messagebox.showinfo(
                "Informazione",
                "L'attacco PGD non è un rumore ambientale da ascoltare direttamente. "
                "Puoi ascoltare l'audio modificato dopo aver premuto 'Modifica audio'."
            )
            return

        if operation not in self.noise_files:
            messagebox.showwarning(
                "Attenzione",
                "Devi prima selezionare un rumore."
            )
            return

        noise_path = self.noise_files[operation]

        if not os.path.exists(noise_path):
            messagebox.showerror(
                "Errore",
                f"File del rumore non trovato:\n{noise_path}"
            )
            return

        self.play_audio(noise_path)

        self.status_label.config(
            text=f"In riproduzione: {operation}",
            fg="black"
        )

    def play_modified_audio(self):
        """
        Riproduce l'audio modificato.

        L'audio modificato può essere:
        - audio originale con rumore ambientale aggiunto;
        - audio modificato tramite attacco adversariale PGD.

        Se l'audio modificato non è ancora stato generato, viene mostrato
        un messaggio di avviso.
        """

        if self.modified_audio_path is None:
            messagebox.showwarning(
                "Attenzione",
                "Devi prima modificare l'audio."
            )
            return

        if not os.path.exists(self.modified_audio_path):
            messagebox.showerror(
                "Errore",
                "Il file audio modificato non esiste."
            )
            return

        self.play_audio(self.modified_audio_path)

        self.status_label.config(
            text="In riproduzione: audio modificato",
            fg="black"
        )

    def play_audio(self, audio_path):
        """
        Riproduce un file audio usando pygame.

        Prima interrompe eventuali riproduzioni già attive,
        poi carica il nuovo file audio e lo manda in riproduzione.
        """

        pygame.mixer.music.stop()
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()

    def stop_audio(self):
        """
        Interrompe la riproduzione audio corrente.

        Aggiorna anche l'etichetta di stato nella GUI.
        """

        pygame.mixer.music.stop()

        self.status_label.config(
            text="Riproduzione interrotta",
            fg="gray"
        )


if __name__ == "__main__":
    """
    Punto di ingresso del programma.

    Crea la finestra principale tkinter, istanzia la classe AudioDeepfakeGUI
    e avvia il loop grafico dell'applicazione.
    """

    root = tk.Tk()
    app = AudioDeepfakeGUI(root)
    root.mainloop()