from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.prompts import MessagesPlaceholder
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import time
import dotenv
from googlesearch import search

dotenv.load_dotenv()

# 웹사이트 메인 헤더
st.title("Horizon Europe Chatbot 🇪🇺")
st.write("이 챗봇은 Horizon Europe 홈페이지의 데이터를 기반으로 질문에 답변합니다.")
st.write("연구 프로젝트의 웹사이트를 알고 싶다면, 예: '0000 프로젝트 웹사이트 알려줘'와 같이 질문하세요.")


# 세션 초기화: 모델과 메시지 저장
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # 대화 히스토리 저장

# 벡터 스토어 로드
vector_store_path = "new_vector_store"
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = FAISS.load_local(vector_store_path, embedding, allow_dangerous_deserialization=True)

# 프롬프트 템플릿 정의
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "당신은 다국어를 지원하는 AI 비서입니다. 질문의 언어가 한국어이면 한국어로 답변하고, 영어이면 영어로 답변하세요. "
        "당신은 주어진 문맥에서만 답변할 수 있습니다. 답변은 자연스럽고 명확한 문장으로 작성하세요. "
        "문맥:\n{context}\n\n"
        "질문: {question}\n\n"
        "답변:"
    )
)

# 구글 검색을 통해 프로젝트 웹사이트 링크 가져오기
def get_project_website(project_name):
    query = f"{project_name} project website"
    search_results = list(search(query, num_results=5))  # 여러 개의 결과를 가져옵니다.
    
    # 검색 결과에서 유효한 웹사이트 링크만 필터링 (도메인 예시: .eu, .com, .org, .edu 등)
    valid_domains = [ ".eu", ".com", ".org", ".edu"]
    for result in search_results:
        if any(domain in result for domain in valid_domains):
            return result  # 유효한 도메인이 발견되면 반환
    return "유효한 웹사이트 링크를 찾을 수 없습니다."

# 이전 대화 히스토리 출력
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(f"{message['content']}")

# 유저 입력 처리
if prompt := st.chat_input("질문을 입력해주세요."):

    # 유저 입력을 화면에 표시하고 세션 상태에 추가
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"{prompt}")

    # 사용자가 "웹사이트" 관련 질문을 했을 때
    if "웹사이트" in prompt:
        # 프로젝트 이름 추출: "웹사이트" 앞의 단어를 프로젝트 이름으로 사용
        project_name = prompt.split("웹사이트")[0].strip()
        website = get_project_website(project_name)
        
        st.session_state["messages"].append({"role": "assistant", "content": f"프로젝트 웹사이트 링크: {website}"})
        with st.chat_message("assistant"):
            st.markdown(f"웹사이트 링크: {website}")
    else:
        # AI 응답 처리
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # 히스토리를 ChatOpenAI에 전달
            llm = ChatOpenAI(model=st.session_state["openai_model"])
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )

            # 대화 히스토리에서 모든 메시지를 유지하고, 최근 메시지를 포함한 질의 생성
            query_with_history = (
                "대화 내용:\n"
                + "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
                + f"\n\n질문: {prompt}"
            )
            
            result = qa_chain({"query": query_with_history})
            full_response = result["result"]

            # 한 글자씩 출력하여 실시간 타이핑 효과 구현
            displayed_text = ""
            for char in full_response:
                displayed_text += char
                message_placeholder.markdown(displayed_text + "▌")  # 커서 효과 추가
                time.sleep(0.02)  # 타이핑 속도 조절 (0.02초 간격)

            message_placeholder.markdown(full_response)  # 최종 결과 표시

        # AI 응답을 세션 상태에 저장
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
