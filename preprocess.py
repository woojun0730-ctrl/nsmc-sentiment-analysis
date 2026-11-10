import os
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    print("=== [1/3] NSMC 공식 원본 데이터 다운로드 ===")
    train_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
    test_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"
    
    if not os.path.exists("ratings_train.txt"):
        print("ratings_train.txt 다운로드 중...")
        urllib.request.urlretrieve(train_url, "ratings_train.txt")
    if not os.path.exists("ratings_test.txt"):
        print("ratings_test.txt 다운로드 중...")
        urllib.request.urlretrieve(test_url, "ratings_test.txt")
        
    train_data = pd.read_table("ratings_train.txt")
    test_data = pd.read_table("ratings_test.txt")
    print(f"다운로드 완료: 공식 Train {len(train_data):,}개, 공식 Test {len(test_data):,}개")

    print("\n=== [2/3] 결측치 및 중복 문장 정제 ===")
    train_data.drop_duplicates(subset=['document'], inplace=True)
    train_data = train_data.dropna(how='any')

    test_data.drop_duplicates(subset=['document'], inplace=True)
    test_data = test_data.dropna(how='any')

    print("\n=== [3/3] 층화 분할 (Train:Valid = 8:2) 및 저장 ===")
    train_df, valid_df = train_test_split(
        train_data, 
        test_size=0.2, 
        random_state=42, 
        stratify=train_data['label']
    )
    test_df = test_data

    print(f"최종 분할 데이터 수:")
    print(f" - Train      : {len(train_df):,}개")
    print(f" - Validation : {len(valid_df):,}개")
    print(f" - Test       : {len(test_df):,}개")

    train_df.to_csv("train_clean.csv", index=False, encoding='utf-8-sig')
    valid_df.to_csv("valid_clean.csv", index=False, encoding='utf-8-sig')
    test_df.to_csv("test_clean.csv", index=False, encoding='utf-8-sig')
    print("정제 파일 저장 완료: train_clean.csv, valid_clean.csv, test_clean.csv")

if __name__ == "__main__":
    main()