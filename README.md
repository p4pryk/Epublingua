# EPUBLingua

**Professional EPUB book translator using LLMs (Large Language Models)**

Translate EPUB books while preserving formatting, structure, and images using Azure OpenAI, OpenAI, or Google Gemini.

**GitHub Repository:** https://github.com/p4pryk/Epublingua

## Features

- Translate EPUB files with perfect formatting preservation
- Multi-threaded processing for fast translation
- Support for Azure OpenAI (default), OpenAI, and Google Gemini
- Preserves images, HTML structure, and formatting
- Progress bar with ETA and statistics
- Configurable target language
- Command-line interface with extensive options

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Azure OpenAI (Default - Recommended)

```bash
# Set environment variables in .env file:
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
# AZURE_OPENAI_KEY=your-key
# AZURE_OPENAI_DEPLOYMENT=gpt-5-mini

python epub_translator.py book.epub
```

### OpenAI

```bash
export OPENAI_API_KEY="your-openai-key"
python epub_translator.py book.epub --provider openai
```

### Google Gemini (Free Tier)

```bash
export GEMINI_API_KEY="your-gemini-key"
python epub_translator.py book.epub --provider gemini
```

## Usage

```
python epub_translator.py INPUT_FILE [options]

Required:
  INPUT_FILE                    Input EPUB file path

Options:
  -o, --output OUTPUT          Output EPUB file (default: INPUT_translated.epub)
  -p, --provider PROVIDER      LLM provider: azure, openai, gemini (default: azure)
  -l, --lang LANGUAGE          Target language (default: Polish)
  -w, --workers N              Number of parallel threads (default: auto - 1 for Gemini, 3 for Azure/OpenAI)
  
Azure OpenAI Options:
  --azure-endpoint URL         Azure OpenAI endpoint (or use AZURE_OPENAI_ENDPOINT env)
  --azure-key KEY              API key (or use AZURE_OPENAI_KEY env)
  --azure-deployment NAME      Deployment name (or use AZURE_OPENAI_DEPLOYMENT env)
  
OpenAI Options:
  --openai-key KEY            OpenAI API key (or use OPENAI_API_KEY env)
  --openai-model MODEL        Model name (default: gpt-4o-mini)
```

## Examples

**Translate to Polish using Azure OpenAI (default):**
```bash
python epub_translator.py book.epub
```

**Translate to English using OpenAI:**
```bash
python epub_translator.py book.epub -l English --provider openai
```

**Translate to Spanish using Gemini:**
```bash
python epub_translator.py book.epub -l Spanish --provider gemini
```

**Use 10 parallel threads for faster translation:**
```bash
python epub_translator.py book.epub -w 10
```

**Specify custom output file:**
```bash
python epub_translator.py book.epub -o translated_book.epub
```

**Translate to French with Azure OpenAI and custom parameters:**
```bash
python epub_translator.py book.epub \
  -l French \
  -w 20 \
  --azure-endpoint "https://your-resource.openai.azure.com/" \
  --azure-key "your-key" \
  --azure-deployment "gpt-5-mini"
```

## How It Works

1. **Extraction**: Unpacks EPUB file (ZIP archive) to temporary directory
2. **Discovery**: Finds all HTML/XHTML content files
3. **Translation**: Translates files in parallel using LLM while preserving HTML structure
4. **Reconstruction**: Packages translated content back into EPUB format
5. **Preservation**: Maintains images, fonts, CSS, and all formatting

## Performance

Translation speed depends on:
- **LLM Provider**: Azure OpenAI is fastest (high rate limits)
- **Number of Workers**: More workers = faster translation (default: 3 for Azure/OpenAI, 1 for Gemini)
- **Book Size**: Larger books take longer
- **Model Speed**: GPT-5-mini, GPT-4o-mini are fast and cost-effective

**Example timing (44-page book):**
- 3 workers: ~6-7 minutes
- 10 workers: ~2-3 minutes
- 20 workers: ~1-2 minutes

## Requirements

- Python 3.8+
- API key for at least one provider:
  - Azure OpenAI (recommended)
  - OpenAI
  - Google Gemini (free tier available)

## Environment Variables

Create a `.env` file in the project directory:

```env
# Azure OpenAI (recommended)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-5-mini

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Google Gemini
GEMINI_API_KEY=your-gemini-api-key
```

## Rate Limits

- **Azure OpenAI**: High limits (1000-5000+ RPM), supports many parallel workers
- **OpenAI**: High limits, supports many parallel workers
- **Google Gemini Free Tier**: 5 requests/minute, automatically limited to 1 worker with 13s delay

## Translation Quality

EPUBLingua uses carefully crafted prompts to ensure:
- Literal translations that preserve meaning
- Technical terms remain in English when appropriate
- HTML structure and formatting perfectly preserved
- Images and alt text handled correctly
- Book layout and pagination maintained

## License

MIT License

## Contributing

Contributions welcome! Please submit pull requests to:
https://github.com/p4pryk/Epublingua

## Support

For issues, questions, or feature requests, please use the GitHub issue tracker:
https://github.com/p4pryk/Epublingua/issues
