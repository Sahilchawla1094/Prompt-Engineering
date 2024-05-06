# Document Processing Application

This Streamlit-based application allows users to upload various types of documents, including PDF, DOCX, TXT, HTML, CSV, and XLSX files, and processes them to extract text or structured data.

## Features

- **Multiple File Formats**: Supports a range of document types for broad applicability.
- **Text Extraction**: Extracts text from PDF, DOCX, and TXT files.
- **Data Handling**: Parses structured data from CSV and XLSX files into a readable and usable format.
- **HTML Parsing**: Extracts and cleans text from HTML files.

## Requirements

- Python 3.6+
- Streamlit
- Pandas
- PyPDF2
- python-docx
- BeautifulSoup
- openpyxl

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Sahilchawla1094/Prompt-Engineering/invitations
   ```
2. Navigate to the project directory:
   ```bash
   cd Prompt-Engineering
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

#### To run the application:
   ```bash
   streamlit run main.py
   ```

Navigate to `http://localhost:8501` in your web browser to see the application in action.

## How It Works

1. ***Upload a File***: Users can upload a file in one of the supported formats.
2. ***File Processing***: Depending on the file extension, the application uses different loaders to process the file:
   - ***.pdf***: Uses PyPDFLoader to extract text from each page.
   - ***.docx***: Uses Docx2txtLoader to extract plain text.
   - ***.txt***: Reads plain text directly.
   - ***.csv***: Loads and displays CSV data.
   - ***.xlsx***: Loads and displays Excel data.
   - ***.html***: Extracts text using BeautifulSoup.
3. ***Display Data***: The extracted data or text is displayed on the webpage.

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.
