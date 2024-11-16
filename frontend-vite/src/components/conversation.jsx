import InputForm from './input-form';
import FeedbackOverlay from "./feedback-overlay";
import Messages from "./messages";
import { useState } from 'react';

export default function Conversation({
  submitPrompt, isResponseGenerating, prompt, setPrompt, messages
}) {
  const [feedback, setFeedback] = useState({});
  const [likedMessages, setLikedMessages] = useState([]);
  const [dislikedMessages, setDislikedMessages] = useState([]);

  return (
    <div className="mt-5">
      {
        Object.keys(feedback).length === 0 ?
          <>
            <Messages dislikedMessages={dislikedMessages} likedMessages={likedMessages}
              messages={messages} setFeedback={setFeedback} isResponseGenerating={isResponseGenerating} />
            <div>
              <div className='h-[10vh] flex items-end justify-center'>
                <div className='w-full md:w-10/12 lg:w-9/12 xl:w-1/2'>
                  <InputForm isResponseGenerating={isResponseGenerating} prompt={prompt}
                    setPrompt={setPrompt} submitPrompt={submitPrompt} />
                </div>
              </div>
            </div>
          </>
        :
          <FeedbackOverlay dislikedMessages={dislikedMessages} setDislikedMessages={setDislikedMessages}
            likedMessages={likedMessages} setLikedMessages={setLikedMessages} feedback={feedback}
            setFeedback={setFeedback} />
      }
    </div>
  );
}
