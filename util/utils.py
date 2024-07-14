import pandas as pd
# 색상 정보를 JSON 파일에서 가져오는 함수 만들기. -> utils.py로 따로 저장해야함.
def get_color_from_json(prdCode):
    # JSON 불러오기
    json_path = 'products.json'
    df = pd.read_json(json_path) 

    # 색상정보, 옵션 1, 옵션2 가져오기
    match_row = df[df['prdCode'] == prdCode]
    if not match_row.empty:
        info = match_row[['color', 'option1', 'option2']]
        print(info)

        hex_color = info['color'].values[0].lstrip('#')
        new_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    else:
        print(f"No Matching prdCode: {prdCode}")
        new_color = (0, 0, 0)
    
    return (new_color[2], new_color[1], new_color[0])  # BGR로 변환
