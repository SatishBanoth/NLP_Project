import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def read_file(self, file_path):
        """Read text from file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def clean_text(self, text):
        """Basic text cleaning"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def remove_stopwords(self, tokens):
        """Remove stopwords from tokenized text"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_text(self, tokens):
        """Apply lemmatization to tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_text(self, tokens):
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_text(self, text):
        """Complete preprocessing pipeline"""
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize_text(tokens)
        tokens = self.stem_text(tokens)
        return tokens

def calculate_cosine_similarity(text1_tokens, text2_tokens):
    """Calculate cosine similarity between two texts"""
    text1 = ' '.join(text1_tokens)
    text2 = ' '.join(text2_tokens)
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return float(cosine_similarity(vectors)[0][1])

def calculate_jaccard_similarity(text1_tokens, text2_tokens):
    """Calculate Jaccard similarity between two texts"""
    set1 = set(text1_tokens)
    set2 = set(text2_tokens)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union if union != 0 else 0)

class TextSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Similarity Analyzer")
        
        # Fixed window size
        window_width = 1200
        window_height = 800
        
        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Calculate position for center of screen
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        
        # Set window size and position
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.root.resizable(False, False)  # Prevent window resizing
        
        # Modern color scheme
        self.colors = {
            'bg': '#1a1a2e',            # Dark blue background
            'primary': '#0f3460',       # Deep blue
            'secondary': '#e94560',     # Coral red
            'accent': '#16213e',        # Midnight blue
            'text': '#ffffff',          # White text
            'text_secondary': '#a4b0be', # Light gray text
            'success': '#4cd137',       # Green
            'drop_zone': '#222831',     # Dark gray
            'drop_zone_hover': '#2d4059' # Slightly lighter gray
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Initialize text processor
        self.processor = TextProcessor()
        
        # File paths
        self.file1_path = None
        self.file2_path = None
        
        self.setup_styles()
        self.setup_ui()
        
    def setup_styles(self):
        style = ttk.Style()
        
        # Main background
        style.configure('Main.TFrame',
                       background=self.colors['bg'])
        
        # Center frame style
        style.configure('Center.TFrame',
                       background=self.colors['bg'])
        
        # Drop zone style
        style.configure('Drop.TFrame', 
                       background=self.colors['drop_zone'],
                       relief='solid',
                       borderwidth=2)
        
        # Hover style for drop zones
        style.configure('DropHover.TFrame',
                       background=self.colors['drop_zone_hover'],
                       relief='solid',
                       borderwidth=2)
        
        # Button style
        style.configure('Action.TButton',
                       padding=(30, 15),
                       font=('Segoe UI', 12, 'bold'))
        
        # Label styles
        style.configure('Title.TLabel',
                       font=('Segoe UI', 32, 'bold'),
                       background=self.colors['bg'],
                       foreground=self.colors['text'],
                       padding=(0, 20))
        
        style.configure('Subtitle.TLabel',
                       font=('Segoe UI', 12),
                       background=self.colors['bg'],
                       foreground=self.colors['text_secondary'])
        
        style.configure('DropZone.TLabel',
                       font=('Segoe UI', 14),
                       background=self.colors['drop_zone'],
                       foreground=self.colors['text'],
                       padding=(0, 20))
        
        style.configure('FileSelected.TLabel',
                       font=('Segoe UI', 11),
                       background=self.colors['drop_zone'],
                       foreground=self.colors['secondary'])
        
        style.configure('Result.TLabel',
                       font=('Segoe UI', 16),
                       background=self.colors['primary'],
                       foreground=self.colors['text'],
                       padding=(20, 10))
        
        # Results frame style
        style.configure('Results.TLabelframe',
                       background=self.colors['primary'])
        
        style.configure('Results.TLabelframe.Label',
                       font=('Segoe UI', 14, 'bold'),
                       background=self.colors['primary'],
                       foreground=self.colors['text'])
        
    def setup_ui(self):
        # Create a center-aligned container
        center_container = ttk.Frame(self.root, style='Center.TFrame')
        center_container.place(relx=0.5, rely=0.5, anchor='center')
        
        # Fixed width for the content area
        content_width = 800
        
        # Main container with fixed width
        main_container = ttk.Frame(center_container, style='Main.TFrame', padding="40")
        main_container.grid(row=0, column=0)
        
        # Title with emoji
        title_label = ttk.Label(main_container, 
                              text="✨ Text Similarity Analyzer ✨",
                              style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 40))
        
        # Subtitle
        subtitle_label = ttk.Label(main_container,
                                 text="Compare the similarity between two text documents",
                                 style='Subtitle.TLabel')
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 40))
        
        # Drop zones container with fixed width
        drop_zones = ttk.Frame(main_container, style='Main.TFrame', width=content_width)
        drop_zones.grid(row=2, column=0, columnspan=2)
        drop_zones.grid_columnconfigure(0, weight=1)
        drop_zones.grid_columnconfigure(1, weight=1)
        
        # First drop zone
        self.drop_zone1 = ttk.Frame(drop_zones, style='Drop.TFrame', padding="40", width=350)
        self.drop_zone1.grid(row=0, column=0, padx=15, sticky='nsew')
        self.drop_zone1.grid_propagate(False)  # Maintain fixed size
        self.drop_zone1.drop_target_register(DND_FILES)
        self.drop_zone1.dnd_bind('<<Drop>>', lambda e: self.drop_file(e, 1))
        
        icon_label1 = ttk.Label(self.drop_zone1,
                              text="📄",
                              font=('Segoe UI', 32),
                              background=self.colors['drop_zone'],
                              foreground=self.colors['text'])
        icon_label1.pack(pady=(10, 0))
        
        self.file1_label = ttk.Label(self.drop_zone1,
                                   text="Drop your first text file here",
                                   style='DropZone.TLabel',
                                   wraplength=250,
                                   justify='center')
        self.file1_label.pack(expand=True)
        
        self.selected_file1_label = ttk.Label(self.drop_zone1,
                                           text="",
                                           style='FileSelected.TLabel')
        self.selected_file1_label.pack(expand=True)
        
        # Second drop zone
        self.drop_zone2 = ttk.Frame(drop_zones, style='Drop.TFrame', padding="40", width=350)
        self.drop_zone2.grid(row=0, column=1, padx=15, sticky='nsew')
        self.drop_zone2.grid_propagate(False)  # Maintain fixed size
        self.drop_zone2.drop_target_register(DND_FILES)
        self.drop_zone2.dnd_bind('<<Drop>>', lambda e: self.drop_file(e, 2))
        
        icon_label2 = ttk.Label(self.drop_zone2,
                              text="📄",
                              font=('Segoe UI', 32),
                              background=self.colors['drop_zone'],
                              foreground=self.colors['text'])
        icon_label2.pack(pady=(10, 0))
        
        self.file2_label = ttk.Label(self.drop_zone2,
                                   text="Drop your second text file here",
                                   style='DropZone.TLabel',
                                   wraplength=250,
                                   justify='center')
        self.file2_label.pack(expand=True)
        
        self.selected_file2_label = ttk.Label(self.drop_zone2,
                                           text="",
                                           style='FileSelected.TLabel')
        self.selected_file2_label.pack(expand=True)
        
        # Compare button
        self.compare_button = ttk.Button(main_container,
                                       text="✨ Compare Texts ✨",
                                       style='Action.TButton',
                                       command=self.compare_texts)
        self.compare_button.grid(row=3, column=0, columnspan=2, pady=40)
        self.compare_button['state'] = 'disabled'
        
        # Results frame
        self.results_frame = ttk.LabelFrame(main_container,
                                          text="Analysis Results",
                                          style='Results.TLabelframe',
                                          padding="30",
                                          width=content_width)
        self.results_frame.grid(row=4, column=0, columnspan=2, sticky='ew')
        
        # Results labels
        self.cosine_label = ttk.Label(self.results_frame,
                                    text="",
                                    style='Result.TLabel')
        self.cosine_label.grid(row=0, column=0, pady=(5, 10), sticky='ew')
        
        self.jaccard_label = ttk.Label(self.results_frame,
                                     text="",
                                     style='Result.TLabel')
        self.jaccard_label.grid(row=1, column=0, pady=(0, 5), sticky='ew')
        
        # Configure grid weights for results frame
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        # Bind events for drop zones
        for zone, widget in [(1, self.drop_zone1), (2, self.drop_zone2)]:
            widget.bind('<Button-1>', lambda e, z=zone: self.select_file(z))
            widget.bind('<Enter>', lambda e, z=zone: self.on_drop_zone_enter(z))
            widget.bind('<Leave>', lambda e, z=zone: self.on_drop_zone_leave(z))
    
    def select_file(self, zone):
        """Handle file selection through dialog"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            self.process_file_selection(file_path, zone)
    
    def drop_file(self, event, zone):
        """Handle file drop"""
        file_path = event.data
        # Remove curly braces if present (Windows)
        file_path = file_path.strip('{}')
        self.process_file_selection(file_path, zone)
    
    def on_drop_zone_enter(self, zone):
        """Handle mouse enter event for drop zones"""
        if zone == 1:
            self.drop_zone1.configure(style='DropHover.TFrame')
        else:
            self.drop_zone2.configure(style='DropHover.TFrame')
    
    def on_drop_zone_leave(self, zone):
        """Handle mouse leave event for drop zones"""
        if zone == 1:
            self.drop_zone1.configure(style='Drop.TFrame')
        else:
            self.drop_zone2.configure(style='Drop.TFrame')
            
    def process_file_selection(self, file_path, zone):
        """Process selected file"""
        if zone == 1:
            self.file1_path = file_path
            self.file1_label.configure(text="Selected File:")
            self.selected_file1_label.configure(
                text=f"📄 {os.path.basename(file_path)}")
        else:
            self.file2_path = file_path
            self.file2_label.configure(text="Selected File:")
            self.selected_file2_label.configure(
                text=f"📄 {os.path.basename(file_path)}")
            
            # Enable compare button if both files are selected
            if self.file1_path and self.file2_path:
                self.compare_button['state'] = 'normal'
            
    def compare_texts(self):
        """Compare the two selected text files"""
        try:
            # Show processing message
            self.cosine_label.configure(text="Processing... Please wait")
            self.jaccard_label.configure(text="")
            self.root.update()
            
            # Read files
            text1 = self.processor.read_file(self.file1_path)
            text2 = self.processor.read_file(self.file2_path)
            
            # Preprocess texts
            text1_tokens = self.processor.preprocess_text(text1)
            text2_tokens = self.processor.preprocess_text(text2)
            
            # Calculate similarities
            cosine_sim = calculate_cosine_similarity(text1_tokens, text2_tokens)
            jaccard_sim = calculate_jaccard_similarity(text1_tokens, text2_tokens)
            
            # Update results with modern formatting
            self.cosine_label.configure(
                text=f"🎯 Cosine Similarity: {cosine_sim:.2%}")
            self.jaccard_label.configure(
                text=f"🎯 Jaccard Similarity: {jaccard_sim:.2%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = TkinterDnD.Tk()
    app = TextSimilarityApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()