import streamlit as st
import streamlit as st




q_n_a = [("first question bla bla bla bla bla bla bla",
                    "first answer") ,
         ("second question", "second answer")]

def run():
    for i, q in enumerate(q_n_a):
        with st.chat_message("user"):
            st.markdown(q[0])
        with st.chat_message("assistant"):
            st.markdown(q[1])



if __name__ == "__main__":
    run()
