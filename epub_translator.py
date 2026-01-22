#!/usr/bin/env python3
"""
EPUBLingua - Professional EPUB book translator using LLMs
Supports Azure OpenAI (default), OpenAI, and Gemini
GitHub: https://github.com/p4pryk/Epublingua
"""
import argparse
import os
import zipfile
import tempfile
import shutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

# Wczytaj zmienne środowiskowe z .env (override=True zapewnia że są na pewno załadowane)
load_dotenv(override=True)

# Separatory dla lepszego UI
SEP = "═" * 70
SEP_THIN = "─" * 70


class EPUBTranslator:
    def __init__(self, provider: str = "azure", target_lang: str = "Polish", 
                 max_workers: int = None, azure_endpoint: str = None, 
                 azure_key: str = None, azure_deployment: str = None,
                 openai_api_key: str = None, openai_model: str = "gpt-4o-mini"):
        """
        Inicjalizacja translatora EPUB
        
        Args:
            provider: "azure" (domyślny), "openai" lub "gemini"
            target_lang: Język docelowy (domyślnie Polish)
            max_workers: Liczba równoległych wątków (auto)
            azure_endpoint: Endpoint Azure OpenAI (lub z ENV: AZURE_OPENAI_ENDPOINT)
            azure_key: Klucz API Azure OpenAI (lub z ENV: AZURE_OPENAI_KEY)
            azure_deployment: Nazwa deploymentu Azure (lub z ENV: AZURE_OPENAI_DEPLOYMENT)
            openai_api_key: Klucz API OpenAI (lub z ENV: OPENAI_API_KEY)
            openai_model: Model OpenAI (domyślnie gpt-4o-mini)
        """
        self.provider = provider
        self.target_lang = target_lang
        
        # Auto-configure workers i delay w zależności od providera
        if max_workers is None:
            # Gemini free tier: 5 requests/minute = 1 worker
            # Azure/OpenAI: wysokie limity = 8 workerów dla szybkiego tłumaczenia
            self.max_workers = 1 if provider == "gemini" else 8
        else:
            self.max_workers = max_workers
            
        # Delay między requestami (w sekundach)
        # Gemini: 13s między requestami dla bezpieczeństwa (5 req/min)
        self.request_delay = 13 if provider == "gemini" else 0
        
        # Timeout dla API (w sekundach) - zapobiega zawieszeniu na dużych plikach
        self.api_timeout = 180  # 3 minuty
        
        # Inicjalizacja klienta LLM
        if provider == "azure":
            # Pobierz z argumentów lub ENV
            endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
            key = azure_key or os.getenv("AZURE_OPENAI_KEY")
            deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            
            if not all([endpoint, key, deployment]):
                raise ValueError("Azure requires: endpoint, key, and deployment name (via args or ENV)")
            
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=key,
                api_version="2024-02-15-preview",
                timeout=self.api_timeout
            )
            self.model = deployment
            
        elif provider == "openai":
            # Standardowy OpenAI
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI requires API key (via --openai-key or OPENAI_API_KEY env)")
            
            self.client = OpenAI(api_key=api_key, timeout=self.api_timeout)
            self.model = openai_model
            
        else:  # gemini - darmowy przez OpenAI library
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Set GEMINI_API_KEY environment variable")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=self.api_timeout
            )
            self.model = "gemini-2.5-flash"
    
    def translate_text(self, html_content: str, max_retries: int = 2) -> str:
        """
        Tłumaczy zawartość HTML zachowując formatowanie
        
        Args:
            html_content: HTML do przetłumaczenia
            max_retries: Liczba prób w przypadku błędu
            
        Returns:
            Przetłumaczony HTML
        """
        if not html_content.strip():
            return html_content
        
        prompt = f"""Translate the following HTML content to {self.target_lang}. 

CRITICAL RULES:
1. Keep ALL HTML tags, attributes, and structure EXACTLY as they are
2. Translate ONLY the text content between tags
3. Keep technical English terms unchanged (e.g., API, server, cloud, etc.)
4. Translate literally - if something doesn't make sense, make it sensible
5. Preserve image tags and alt attributes (translate alt text if present)
6. Maintain exact formatting, spacing, and line breaks
7. Return ONLY the translated HTML, no explanations

HTML to translate:
{html_content}"""

        for attempt in range(max_retries + 1):
            try:
                # Przygotuj parametry dla API
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": f"You are a professional translator to {self.target_lang}. You preserve HTML structure perfectly."},
                        {"role": "user", "content": prompt}
                    ]
                }
                
                # Nowsze modele (gpt-4o, gpt-5) używają max_completion_tokens i nie wspierają temperature
                if "gpt-5" in self.model.lower() or "gpt-4o" in self.model.lower():
                    api_params["max_completion_tokens"] = 16000  # Zwiększone dla dużych plików
                    # gpt-5 i gpt-4o nie wspierają custom temperature - używają domyślnej (1)
                else:
                    api_params["max_tokens"] = 16000
                    api_params["temperature"] = 0.3
                
                response = self.client.chat.completions.create(**api_params)
                
                translated = response.choices[0].message.content.strip()
                
                # Usuń ewentualne markdown code blocks
                if translated.startswith("```"):
                    lines = translated.split('\n')
                    translated = '\n'.join(lines[1:-1]) if len(lines) > 2 else translated
                
                # Rate limiting - opóźnienie między requestami
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                    
                return translated
                
            except Exception as e:
                if attempt < max_retries:
                    tqdm.write(f"\n[RETRY] Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    # Use tqdm.write to not break progress bar
                    tqdm.write(f"\n[ERROR] Translation failed after {max_retries + 1} attempts: {e}")
                    return html_content  # Return original on error
        
        return html_content
    
    def extract_epub(self, epub_path: str, extract_dir: str):
        """Wypakowuje EPUB do katalogu"""
        with zipfile.ZipFile(epub_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    def create_epub(self, source_dir: str, output_path: str):
        """Tworzy EPUB z katalogu"""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # mimetype musi być pierwszy i nieskompresowany
            mimetype_path = Path(source_dir) / 'mimetype'
            if mimetype_path.exists():
                zipf.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
            
            # Dodaj resztę plików
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    if file == 'mimetype':
                        continue
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)
    
    def _translate_body_in_chunks(self, body, max_chunk_size: int) -> str:
        """
        Tłumaczy duże body dzieląc je na mniejsze fragmenty.
        Rekurencyjnie wchodzi w duże elementy (div, section) by znaleźć mniejsze części.
        
        Args:
            body: BeautifulSoup element body
            max_chunk_size: Maksymalny rozmiar jednego chunka
            
        Returns:
            Przetłumaczone body jako string HTML
        """
        from bs4 import NavigableString, Tag
        
        def get_chunkable_children(element, depth=0):
            """
            Pobiera dzieci elementu do podziału.
            Jeśli element ma tylko jedno duże dziecko (div/section), wchodzi głębiej.
            """
            # Pobierz tylko znaczące dzieci (tagi i niepuste teksty)
            children = []
            for c in element.children:
                if isinstance(c, Tag):
                    children.append(c)
                elif isinstance(c, NavigableString) and str(c).strip():
                    children.append(c)
            
            # Znajdź duże kontenery (div, section, etc.)
            large_containers = [c for c in children 
                               if isinstance(c, Tag) 
                               and c.name in ('div', 'section', 'article', 'main')
                               and len(str(c)) > max_chunk_size]
            
            # Jeśli mamy dokładnie 1 duży kontener i reszta to małe elementy/whitespace
            if len(large_containers) == 1 and depth < 3:
                container = large_containers[0]
                other_children = [c for c in children if c != container]
                other_size = sum(len(str(c)) for c in other_children)
                
                # Jeśli inne elementy są małe, wejdź w duży kontener
                if other_size < max_chunk_size:
                    tqdm.write(f"    Entering large <{container.name}> ({len(str(container)):,} chars) to find smaller chunks...")
                    inner_children, inner_wrapper = get_chunkable_children(container, depth + 1)
                    
                    # Zwróć dzieci kontenera + wrapper info
                    if inner_wrapper:
                        return inner_children, inner_wrapper
                    else:
                        return inner_children, container
            
            return children, None
        
        # Znajdź elementy do podziału (może wejść głębiej w strukturę)
        children, wrapper_element = get_chunkable_children(body)
        
        # Podziel dzieci na chunki
        chunks = []
        current_chunk = []
        current_size = 0
        
        for child in children:
            child_str = str(child)
            child_size = len(child_str)
            
            # Jeśli pojedynczy element jest większy niż max_chunk_size
            if child_size > max_chunk_size:
                # Zapisz dotychczasowy chunk jeśli niepusty
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0
                # Dodaj duży element jako osobny chunk (nie możemy go dalej dzielić)
                chunks.append([child])
            elif current_size + child_size > max_chunk_size:
                # Zapisz dotychczasowy chunk i rozpocznij nowy
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [child]
                current_size = child_size
            else:
                # Dodaj do bieżącego chunka
                current_chunk.append(child)
                current_size += child_size
        
        # Dodaj ostatni chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        tqdm.write(f"    Split into {len(chunks)} chunks")
        
        # Tłumacz każdy chunk
        translated_parts = []
        for i, chunk in enumerate(chunks, 1):
            chunk_html = ''.join(str(c) for c in chunk)
            tqdm.write(f"    Translating chunk {i}/{len(chunks)} ({len(chunk_html):,} chars)...")
            translated_chunk = self.translate_text(chunk_html)
            translated_parts.append(translated_chunk)
        
        # Złóż z powrotem - uwzględnij wrapper jeśli był
        translated_content = "".join(translated_parts)
        
        if wrapper_element:
            # Odbuduj wrapper (np. div) z atrybutami
            wrapper_attrs = ' '.join(f'{k}="{v}"' if isinstance(v, str) else f'{k}="{" ".join(v)}"' 
                                    for k, v in wrapper_element.attrs.items())
            if wrapper_attrs:
                translated_content = f'<{wrapper_element.name} {wrapper_attrs}>{translated_content}</{wrapper_element.name}>'
            else:
                translated_content = f'<{wrapper_element.name}>{translated_content}</{wrapper_element.name}>'
        
        # Złóż w body
        body_attrs = ' '.join(f'{k}="{v}"' if isinstance(v, str) else f'{k}="{" ".join(v)}"' 
                             for k, v in body.attrs.items())
        if body_attrs:
            return f'<body {body_attrs}>{translated_content}</body>'
        else:
            return f'<body>{translated_content}</body>'
    
    def get_html_files(self, extract_dir: str) -> List[Path]:
        """Znajduje wszystkie pliki HTML/XHTML w EPUB"""
        html_files = []
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.endswith(('.html', '.xhtml', '.htm')):
                    html_files.append(Path(root) / file)
        return sorted(html_files)
    
    def translate_html_file(self, file_path: Path) -> Tuple[Path, bool]:
        """
        Tłumaczy pojedynczy plik HTML
        
        Returns:
            Tuple (file_path, success)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Tłumacz zawartość body (zachowując strukturę)
            body = soup.find('body')
            if body:
                # Zachowaj oryginalne atrybuty body
                original_body_attrs = dict(body.attrs)
                body_str = str(body)
                
                # Sprawdź rozmiar body - jeśli > 30k znaków, użyj chunkingu
                MAX_CHUNK_SIZE = 30000
                
                if len(body_str) > MAX_CHUNK_SIZE:
                    # Duży plik - podziel na chunki i tłumacz osobno
                    tqdm.write(f"  [CHUNK] {file_path.name} is large ({len(body_str):,} chars), splitting...")
                    translated_body = self._translate_body_in_chunks(body, MAX_CHUNK_SIZE)
                else:
                    # Mały plik - tłumacz w całości
                    translated_body = self.translate_text(body_str)
                
                # Parsuj przetłumaczoną zawartość
                translated_soup = BeautifulSoup(translated_body, 'html.parser')
                
                # Sprawdź czy API zwróciło pełny tag <body> czy tylko zawartość
                new_body = translated_soup.find('body')
                
                if new_body:
                    # API zwróciło pełny <body>...</body> - podmień całe body
                    # Przywróć oryginalne atrybuty jeśli zostały utracone
                    for attr, value in original_body_attrs.items():
                        if attr not in new_body.attrs:
                            new_body[attr] = value
                    body.replace_with(new_body)
                else:
                    # API zwróciło tylko zawartość (bez tagu body)
                    # Wyczyść body i wstaw nową zawartość zachowując tag body
                    body.clear()
                    # Wstaw każdy element z przetłumaczonej zawartości
                    for element in list(translated_soup.children):
                        if hasattr(element, 'extract'):
                            body.append(element.extract())
                        else:
                            body.append(str(element))
            
            # Save translated file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(soup))
            
            return (file_path, True)
            
        except Exception as e:
            # Use tqdm.write to not break progress bar
            tqdm.write(f"\n[ERROR] Failed to process {file_path.name}: {e}")
            return (file_path, False)
    
    def translate_epub(self, input_path: str, output_path: str):
        """
        Główna funkcja tłumacząca EPUB
        
        Args:
            input_path: Ścieżka do EPUB wejściowego
            output_path: Path to output EPUB file
        """
        print(f"\n{SEP}")
        print(f"  EPUBLINGUA - EPUB Book Translator")
        print(f"  https://github.com/p4pryk/Epublingua")
        print(f"{SEP_THIN}")
        print(f"  Provider:       {self.provider.upper()} ({self.model})")
        print(f"  Input file:     {Path(input_path).name}")
        print(f"  Target lang:    {self.target_lang}")
        print(f"  Parallel tasks: {self.max_workers} threads")
        print(f"{SEP}\n")
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            print("[1/3] Extracting EPUB archive...")
            self.extract_epub(input_path, temp_dir)
            
            # Find HTML files
            html_files = self.get_html_files(temp_dir)
            print(f"      Found {len(html_files)} HTML file(s) to translate\n")
            
            if not html_files:
                print(f"\n{SEP}")
                print("ERROR: No HTML files found in EPUB")
                print(f"{SEP}\n")
                return
            
            # Translate in parallel
            print("[2/3] Starting translation...\n")
            start_time = time.time()
            failed_files = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.translate_html_file, f): f 
                          for f in html_files}
                
                # Progress bar - jeden, stały
                with tqdm(total=len(html_files), desc="Translating", unit="page", 
                         position=0, leave=True, dynamic_ncols=False, ncols=100,
                         bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                    for future in as_completed(futures):
                        file_path, success = future.result()
                        if not success:
                            failed_files.append(file_path.name)
                            tqdm.write(f"⚠️  Failed: {file_path.name}")
                        pbar.update(1)
            
            elapsed_time = time.time() - start_time
            
            # Create translated EPUB
            print(f"\n[3/3] Creating translated EPUB...")
            self.create_epub(temp_dir, output_path)
            
            # Summary
            print(f"\n{SEP}")
            print(f"  TRANSLATION COMPLETED")
            print(f"{SEP_THIN}")
            print(f"  Output file:    {Path(output_path).name}")
            print(f"  Pages:          {len(html_files) - len(failed_files)}/{len(html_files)} successful")
            print(f"  Time elapsed:   {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
            print(f"  Avg per page:   {elapsed_time/len(html_files):.1f}s")
            if failed_files:
                print(f"  [!] Failed:     {len(failed_files)} file(s)")
            print(f"{SEP}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Translate EPUB books using LLMs (Azure OpenAI, OpenAI, or Gemini)"
    )
    parser.add_argument("input", help="Input EPUB file")
    parser.add_argument("-o", "--output", help="Output EPUB file (default: input_translated.epub)")
    parser.add_argument("-p", "--provider", choices=["azure", "openai", "gemini"], 
                       default="azure", help="LLM provider (default: azure)")
    parser.add_argument("-l", "--lang", default="Polish", 
                       help="Target language (default: Polish)")
    parser.add_argument("-w", "--workers", type=int, default=None,
                       help="Number of parallel threads (default: auto - 1 for Gemini, 3 for Azure/OpenAI)")
    
    # Azure OpenAI options
    parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint (or use AZURE_OPENAI_ENDPOINT env)")
    parser.add_argument("--azure-key", help="Azure OpenAI API key (or use AZURE_OPENAI_KEY env)")
    parser.add_argument("--azure-deployment", help="Azure OpenAI deployment name (or use AZURE_OPENAI_DEPLOYMENT env)")
    
    # OpenAI options
    parser.add_argument("--openai-key", help="OpenAI API key (or use OPENAI_API_KEY env)")
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    
    args = parser.parse_args()
    
    # Ustal output path
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_translated.epub")
    
    # Check if file exists
    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        return
    
    # Create translator
    try:
        translator = EPUBTranslator(
            provider=args.provider,
            target_lang=args.lang,
            max_workers=args.workers,
            azure_endpoint=args.azure_endpoint,
            azure_key=args.azure_key,
            azure_deployment=args.azure_deployment,
            openai_api_key=args.openai_key,
            openai_model=args.openai_model
        )
        
        # Translate
        translator.translate_epub(args.input, args.output)
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
