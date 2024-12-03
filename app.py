import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# OpenAI API 키 설정
openai_api_key = "your api key"

# 벡터 스토어 경로
vector_store_path = "vector_store"

# 벡터 스토어 로드
def load_vector_store(path):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


# RAG 체인 빌드
def build_rag_chain(vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    chat_model = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

    # 프롬프트 정의
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an AI assistant. Use the provided context to answer the question. "
            "If the context does not have the answer, state that the information is not available.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
    )

    return retrieval_qa_chain

# Streamlit UI 구성
def main():
    st.title("Horizon Europe Chatbot")
    st.write("이 챗봇은 Horizon Europe 홈페이지의 데이터를 기반으로 영어질문에 답변합니다.")

    # 벡터 스토어 로드
    st.sidebar.header("Vector Store Status")
    try:
        vector_store = load_vector_store(vector_store_path)
        st.sidebar.success("Vector store loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Failed to load vector store: {e}")
        return

    # RAG 체인 생성
    rag_chain = build_rag_chain(vector_store)

    # 사용자 입력
    user_input = st.text_input("Question:", "")
    if user_input:
        with st.spinner("Processing your query..."):
            try:
                response = rag_chain({"query": user_input})
                st.success("Answer:")
                st.write(response["result"])
            except Exception as e:
                st.error(f"Error processing your query: {e}")

    # 검색된 관련 문서 보기
    if user_input:
        with st.expander("Searched documents"):
            relevant_docs = rag_chain.retriever.get_relevant_documents(user_input)
            for i, doc in enumerate(relevant_docs):
                st.write(f"**문서 {i + 1}:**")
                st.write(doc.page_content)

if __name__ == "__main__":
    main()
