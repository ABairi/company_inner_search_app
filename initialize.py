"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from docx import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    # 指定のログフォルダが存在すれば読み込み、存在しなければ新規作成
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    # 引数に指定した名前のロガー（ログを記録するオブジェクト）を取得
    # 再度別の箇所で呼び出した場合、すでに同じ名前のロガーが存在していれば読み込む
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにロガーにハンドラー（ログの出力先を制御するもの）が設定されている場合、同じログ出力が複数回行われないよう処理を中断する
    if logger.hasHandlers():
        return

    # 1日単位でログファイルの中身をリセットし、切り替える設定
    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    # 出力するログメッセージのフォーマット定義
    # - 「levelname」: ログの重要度（INFO, WARNING, ERRORなど）
    # - 「asctime」: ログのタイムスタンプ（いつ記録されたか）
    # - 「lineno」: ログが出力されたファイルの行番号
    # - 「funcName」: ログが出力された関数名
    # - 「session_id」: セッションID（誰のアプリ操作か分かるように）
    # - 「message」: ログメッセージ
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )

    # 定義したフォーマッターの適用
    log_handler.setFormatter(formatter)

    # ログレベルを「INFO」に設定
    logger.setLevel(logging.INFO)

    # 作成したハンドラー（ログ出力先を制御するオブジェクト）を、
    # ロガー（ログメッセージを実際に生成するオブジェクト）に追加してログ出力の最終設定
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        # ランダムな文字列（セッションID）を、ログ出力用に作成
        st.session_state.session_id = uuid4().hex


def initialize_retriever():
    """
    画面読み込み時にRAGのRetriever（ベクターストアから検索するオブジェクト）を作成
    """
    # ロガーを読み込むことで、後続の処理中に発生したエラーなどがログファイルに記録される
    logger = logging.getLogger(ct.LOGGER_NAME)

    # すでにRetrieverが作成済みの場合、後続の処理を中断
    if "retriever" in st.session_state:
        return
    
    # RAGの参照先となるデータソースの読み込み
    docs_all = load_data_sources()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    # 埋め込みモデルの用意
    embeddings = OpenAIEmbeddings()
    
    # CSVファイルとその他のファイルを分離・統合処理
    processed_docs = []
    csv_docs_by_file = {}
    
    for doc in docs_all:
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            file_extension = os.path.splitext(doc.metadata['source'])[1].lower()
            if file_extension == '.csv':
                # CSVファイルごとにドキュメントをグループ化
                file_path = doc.metadata['source']
                if file_path not in csv_docs_by_file:
                    csv_docs_by_file[file_path] = []
                csv_docs_by_file[file_path].append(doc)
            else:
                processed_docs.append(doc)
        else:
            processed_docs.append(doc)
    
    # CSVファイルごとに統合・最適化処理を実行
    for file_path, csv_docs in csv_docs_by_file.items():
        enhanced_doc = create_enhanced_csv_document(csv_docs, file_path)
        processed_docs.append(enhanced_doc)
    
    # チャンク分割用のオブジェクトを作成
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n"
    )

    # CSVファイル以外のドキュメントのみチャンク分割を実施
    non_csv_docs = [doc for doc in processed_docs if not (hasattr(doc, 'metadata') and doc.metadata.get('file_type') == 'enhanced_csv')]
    csv_enhanced_docs = [doc for doc in processed_docs if hasattr(doc, 'metadata') and doc.metadata.get('file_type') == 'enhanced_csv']
    
    splitted_docs = text_splitter.split_documents(non_csv_docs)
    
    # 統合されたCSVドキュメントは分割せずに追加
    all_docs = splitted_docs + csv_enhanced_docs

    # ベクターストアの作成
    db = Chroma.from_documents(all_docs, embedding=embeddings)

    # ベクターストアを検索するRetrieverの作成
    st.session_state.retriever = db.as_retriever(search_kwargs={"k": ct.RAG_K})


def create_enhanced_csv_document(csv_docs, file_path):
    """
    CSVドキュメントを統合し、検索精度を向上させる
    
    Args:
        csv_docs: 同一CSVファイルからのドキュメントリスト
        file_path: CSVファイルのパス
    
    Returns:
        検索最適化されたドキュメント
    """
    import copy
    
    # 最初のドキュメントをベースに使用
    base_doc = copy.deepcopy(csv_docs[0])
    
    # 全ドキュメントの内容を結合
    combined_content = ""
    for doc in csv_docs:
        combined_content += doc.page_content + "\n"
    
    # CSVの内容を解析
    lines = combined_content.strip().split('\n')
    if len(lines) < 2:
        return base_doc
    
    # ヘッダー行を取得
    header_line = lines[0]
    columns = [col.strip().strip('"') for col in header_line.split(',')]
    
    # データ行を取得
    data_rows = lines[1:]
    
    # ファイル情報
    file_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    
    # 検索最適化されたコンテンツを作成
    enhanced_content = f"""ファイル名: {file_name}
データ種別: CSV表形式データ
総行数: {len(data_rows)}行
カラム数: {len(columns)}個

=== データ構造 ===
カラム一覧: {', '.join(columns)}

=== 検索用キーワード ===
ファイル関連: {file_name_without_ext}, CSV, 表, データ, 一覧
カラム関連: {', '.join(columns)}

=== データ内容 ===
"""
    
    # 各行のデータを検索しやすい形式に変換
    all_values = set()
    for i, row in enumerate(data_rows):
        cells = [cell.strip().strip('"') for cell in row.split(',')]
        enhanced_content += f"\n--- 行{i+2} ---\n"
        
        # カラム名:値のペア形式で追加
        for column, cell in zip(columns, cells):
            if cell:  # 空でない値のみ
                enhanced_content += f"{column}: {cell}\n"
                all_values.add(cell)
        
        # 行全体を一文でも表現
        enhanced_content += f"この行の内容: {', '.join([f'{col}は{val}' for col, val in zip(columns, cells) if val])}\n"
    
    # 全値をキーワードとして追加
    unique_values = list(all_values)
    enhanced_content += f"\n=== データに含まれる全ての値（検索用） ===\n{', '.join(unique_values)}\n"
    
    # よくある検索パターンに対応
    enhanced_content += f"\n=== 検索パターン対応 ===\n"
    enhanced_content += f"このファイルには以下の情報が含まれています:\n"
    for column in columns:
        column_values = []
        for row in data_rows:
            cells = [cell.strip().strip('"') for cell in row.split(',')]
            if len(cells) > columns.index(column):
                val = cells[columns.index(column)]
                if val and val not in column_values:
                    column_values.append(val)
        if column_values:
            enhanced_content += f"- {column}: {', '.join(column_values[:10])}{'...' if len(column_values) > 10 else ''}\n"
    
    # 統計情報
    enhanced_content += f"\n=== 統計情報 ===\n"
    enhanced_content += f"データ行数: {len(data_rows)}行\n"
    enhanced_content += f"項目数: {len(columns)}項目\n"
    
    # 元のCSVデータも保持
    enhanced_content += f"\n=== 元データ ===\n{combined_content}"
    
    # ドキュメントを更新
    base_doc.page_content = enhanced_content
    base_doc.metadata['file_type'] = 'enhanced_csv'
    base_doc.metadata['columns'] = ', '.join(columns)
    base_doc.metadata['row_count'] = len(data_rows)
    base_doc.metadata['all_values'] = ', '.join(unique_values)
    base_doc.metadata['keywords'] = f"{file_name_without_ext}, {', '.join(columns)}, {', '.join(unique_values[:50])}"
    
    return base_doc


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        # 「表示用」の会話ログを順次格納するリストを用意
        st.session_state.messages = []
        # 「LLMとのやりとり用」の会話ログを順次格納するリストを用意
        st.session_state.chat_history = []


def load_data_sources():
    """
    RAGの参照先となるデータソースの読み込み

    Returns:
        読み込んだ通常データソース
    """
    # データソースを格納する用のリスト
    docs_all = []
    # ファイル読み込みの実行（渡した各リストにデータが格納される）
    recursive_file_check(ct.RAG_TOP_FOLDER_PATH, docs_all)

    web_docs_all = []
    # ファイルとは別に、指定のWebページ内のデータも読み込み
    # 読み込み対象のWebページ一覧に対して処理
    for web_url in ct.WEB_URL_LOAD_TARGETS:
        # 指定のWebページを読み込み
        loader = WebBaseLoader(web_url)
        web_docs = loader.load()
        # for文の外のリストに読み込んだデータソースを追加
        web_docs_all.extend(web_docs)
    # 通常読み込みのデータソースにWebページのデータを追加
    docs_all.extend(web_docs_all)

    return docs_all


def recursive_file_check(path, docs_all):
    """
    RAGの参照先となるデータソースの読み込み

    Args:
        path: 読み込み対象のファイル/フォルダのパス
        docs_all: データソースを格納する用のリスト
    """
    # パスがフォルダかどうかを確認
    if os.path.isdir(path):
        # フォルダの場合、フォルダ内のファイル/フォルダ名の一覧を取得
        files = os.listdir(path)
        # 各ファイル/フォルダに対して処理
        for file in files:
            # ファイル/フォルダ名だけでなく、フルパスを取得
            full_path = os.path.join(path, file)
            # フルパスを渡し、再帰的にファイル読み込みの関数を実行
            recursive_file_check(full_path, docs_all)
    else:
        # パスがファイルの場合、ファイル読み込み
        file_load(path, docs_all)


def file_load(path, docs_all):
    """
    ファイル内のデータ読み込み

    Args:
        path: ファイルパス
        docs_all: データソースを格納する用のリスト
    """
    # ファイルの拡張子を取得
    file_extension = os.path.splitext(path)[1]
    # ファイル名（拡張子を含む）を取得
    file_name = os.path.basename(path)

    # 想定していたファイル形式の場合のみ読み込む
    if file_extension in ct.SUPPORTED_EXTENSIONS:
        # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
        loader = ct.SUPPORTED_EXTENSIONS[file_extension](path)
        docs = loader.load()
        docs_all.extend(docs)


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s