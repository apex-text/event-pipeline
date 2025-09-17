# src/natural_language_query/convert_to_json_array.py
import json
import os

def convert_jsonl_to_json_array(input_path, output_path):
    """
    JSON Lines (.jsonl) 파일을 읽어 표준 JSON 배열 (.json) 파일로 변환합니다.
    """
    print(f"Starting conversion: {input_path} -> {output_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        print("Please make sure the input file is in the same directory as this script.")
        return

    records = []
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                # 빈 줄은 건너뜁니다.
                if line.strip():
                    records.append(json.loads(line))
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(records, outfile, indent=4, ensure_ascii=False)
            
        print(f"Successfully converted {len(records)} records.")
        print(f"Output file is ready: '{output_path}'")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from input file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # 이 스크립트가 있는 디렉토리를 기준으로 파일 경로를 설정합니다.
    script_dir = os.path.dirname(__file__)
    
    # --- 사용자 설정 ---
    # MinIO에서 다운로드한 원본 파일명을 여기에 입력하세요.
    # 예: 'part-00000-....json'
    input_filename = "input_sample.jsonl" 
    
    # 생성될 최종 JSON 파일명입니다.
    output_filename = "gdelt_sample_for_upload.json"
    # --------------------

    input_filepath = os.path.join(script_dir, input_filename)
    output_filepath = os.path.join(script_dir, output_filename)
    
    print("--- JSONL to JSON Array Converter ---")
    print(f"NOTE: Please rename your downloaded file to '{input_filename}' and place it in the same directory as this script.")
    
    convert_jsonl_to_json_array(input_filepath, output_filepath)
