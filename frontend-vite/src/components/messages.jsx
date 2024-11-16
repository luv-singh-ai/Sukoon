import {useEffect, useRef, useState} from 'react';
import ThumbsUpOutline from "../assets/thumbs-up-outline.svg";
import ThumbsUpFilled from "../assets/thumbs-up-filled.svg";
import ThumbsDownFilled from "../assets/thumbs-down-filled.svg";
import ThumbsDownOutline from "../assets/thumbs-down-outline.svg";
import ChatLoadingAnimation from './chat-loading-animation/chat-loading-animation';

export default function Messages({
  messages, setFeedback, likedMessages, dislikedMessages, isResponseGenerating
}) {
  const lastMessageRef = useRef(null);

  // Scroll to the last message whenever messages change
  useEffect(() => {
    if (lastMessageRef.current) {
      lastMessageRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="h-full px-[6vw] sm:px-[8vw] lg:px-[10vw] xl:px-[25vw] overflow-y-scroll">
      {
        messages.map((message, i) => {
          if (message.isResponse !== true) {
            return (
              <div key={i} style={{backgroundColor: "rgba(255, 229, 180, 0.5)"}}
                className="mt-5 max-w-60 sm:max-w-sm md:max-w-md lg:max-w-lg xl:max-w-xl p-2 px-5 mb-5 rounded-l-3xl rounded-y-3xl rounded-tr-3xl clear-both float-right">
                <p className="text-sm md:text-md text-wrap">{message.message}</p>
              </div>
            );
          } else {
            return (
              <div key={i}>
                <div style={{backgroundColor: "rgba(255, 229, 180, 0.5)"}}
                  className="relative w-5/6 xl:w-2/3 p-2 px-5 mb-5 rounded-r-3xl rounded-y-3xl rounded-tl-3xl clear-both">
                  <p className="text-sm md:text-md text-wrap">{message.message}</p>
                  <img src={likedMessages.includes(i) ? ThumbsUpFilled : ThumbsUpOutline}
                    onClick={() => {
                      if (likedMessages.includes(i) === false) {
                        setFeedback({feedback: "like", messageId: i});
                      }
                    }} className="absolute w-5 xl:w-6 bottom-[-26px] xl:bottom-[-28px] cursor-pointer" />
                  <img src={dislikedMessages.includes(i) ? ThumbsDownFilled : ThumbsDownOutline}
                    onClick={() => {
                      if (dislikedMessages.includes(i) === false) {
                        setFeedback({feedback: "dislike", messageId: i});
                      }
                    }} className="absolute w-5 xl:w-6 bottom-[-26px] xl:bottom-[-28px] left-[60px] cursor-pointer" />
                </div>
              </div>
            );
          }
        })
      }
      {
        isResponseGenerating && <div style={{backgroundColor: "rgba(255, 229, 180, 0.5)"}}
          className="w-fit px-5 mb-5 rounded-r-3xl rounded-y-3xl rounded-tl-3xl clear-both">
          <ChatLoadingAnimation />
        </div>
      }
      <div ref={lastMessageRef} />
    </div>
  )
}
