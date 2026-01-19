import streamlit as st
import tempfile
import os
import traceback 

# [변경 1] 환경 변수 로드를 위한 라이브러리 임포트
from dotenv import load_dotenv

# LangChain 관련 모듈 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# [변경 2] .env 파일 활성화 (로컬 개발 시 .env 파일에서 키를 읽어옴)
load_dotenv()

# 1. 페이지 기본 설정
st.set_page_config(page_title="나만의 RAG 챗봇", page_icon="🐻")
st.title("🐻 PDF 기반 RAG 챗봇")

# [변경 3] API KEY 입력창 제거 -> 환경 변수에서 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

# API Key가 없는 경우 경고 표시 및 중단
if not openai_api_key:
    st.error("환경 변수 `OPENAI_API_KEY`가 설정되지 않았습니다. .env 파일이나 시스템 환경 변수를 확인해주세요.")
    st.stop()  # 키가 없으면 앱 실행을 여기서 멈춤

st.markdown("---")

# 사이드바: 설정 및 입력
with st.sidebar:
    st.header("설정 (Configuration)")
    
    # 2. 문서 업로드 및 카테고리 선택
    st.subheader("문서 업로드 & 선택")
    uploaded_files = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"], accept_multiple_files=True)
    
    # 업로드된 파일이 있을 경우 선택 박스 활성화
    selected_doc = None
    if uploaded_files:
        doc_names = [file.name for file in uploaded_files]
        selected_doc_name = st.selectbox("검색할 문서를 선택하세요 (카테고리)", doc_names)
        
        # 선택된 파일 객체 찾기
        for file in uploaded_files:
            if file.name == selected_doc_name:
                selected_doc = file
                break
    
    st.markdown("---")
    
    # 5. 시스템 프롬프트 설정 (사용자 입력 가능)
    st.subheader("시스템 프롬프트 설정")
    default_system_prompt = """당신은 질문에 답변하는 작업을 수행하는 친절한 어시스턴트입니다.
다음에 제공된 문맥 정보를 바탕으로 질문에 답하세요.
정답을 모를 경우, 모른다고만 말하세요.
답변은 반드시 한국어로 작성하세요."""
    
    system_prompt_input = st.text_area("AI에게 부여할 역할/지시사항", value=default_system_prompt, height=200)
    
    process_btn = st.button("문서 처리 및 챗봇 초기화")

# 세션 상태 초기화 (채팅 기록, 벡터 저장소 등)
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# --- LangChain 구성 요소 함수화 (모듈 분리) ---

def process_pdf(file):
    """
    [Document Load] Streamlit 업로드 파일을 임시 저장 후 PyPDFLoader로 로드
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name
        
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path) # 임시 파일 삭제
    return docs

def split_text(docs):
    """
    [Text Split] 문서를 청크 단위로 분할
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=300, 
        chunk_overlap=50
    )
    return text_splitter.split_documents(docs)

def create_vectorstore(chunks):
    """
    [Embedding & VectorStore] 임베딩 생성 및 FAISS 저장소 구축
    [변경 4] openai_api_key 인자 제거 (LangChain이 환경변수를 자동 인식함, 혹은 전역 변수 사용)
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", 
        openai_api_key=openai_api_key  # 전역 변수 혹은 환경변수 사용
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def get_rag_chain(vectorstore, system_prompt):
    """
    [Chain] Retriever, Prompt, LLM 연결
    [변경 5] api_key 인자 제거
    """
    # 1. Retriever 설정 (MMR 방식)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.8}
    )

    # 2. Prompt Template 설정
    template = system_prompt + "\n\n#문맥:\n{context}\n\n#질문:\n{question}\n\n#답변:"
    prompt = PromptTemplate.from_template(template)

    # 3. LLM 설정
    llm = ChatOpenAI(
        model_name="gpt-4o-mini", 
        temperature=0,
        openai_api_key=openai_api_key # 전역 변수 사용
    )

    # 4. Chain 구성 (LCEL 문법)
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- 메인 로직 실행 ---

# 버튼이 눌리면 문서 처리 시작
if process_btn:
    # [변경 6] API Key 유효성 검사 로직 제거 (위에서 st.stop으로 이미 처리됨)
    if not selected_doc:
        st.error("문서를 업로드하고 선택해주세요.")
    else:
        with st.spinner(f"'{selected_doc.name}' 문서를 분석 중입니다..."):
            try:
                # 1. Document Load
                raw_docs = process_pdf(selected_doc)
                # 2. Text Split
                chunks = split_text(raw_docs)
                # 3. Embedding & VectorStore
                # [변경 7] 인자 전달 방식 간소화
                vectorstore = create_vectorstore(chunks)
                
                # 세션에 저장
                st.session_state["vectorstore"] = vectorstore
                
                # 채팅 기록 초기화
                st.session_state["messages"] = [{"role": "assistant", "content": "문서 분석이 완료되었습니다! 질문해주세요."}]
                st.success("완료!")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                st.code(traceback.format_exc())

# 채팅 인터페이스
# 1. 이전 대화 출력
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 2. 사용자 입력 처리
if query := st.chat_input("질문을 입력하세요..."):
    # 사용자 질문 표시
    st.session_state["messages"].append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    # 답변 생성
    if st.session_state["vectorstore"] is not None:
        try:
            # Chain 생성
            # [변경 8] 인자 전달 방식 간소화
            rag_chain = get_rag_chain(
                st.session_state["vectorstore"], 
                system_prompt_input
            )
            
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    response = rag_chain.invoke(query)
                    st.write(response)
            
            # 답변 저장
            st.session_state["messages"].append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
    else:
        st.warning("먼저 문서를 업로드하고 '초기화' 버튼을 눌러주세요.")