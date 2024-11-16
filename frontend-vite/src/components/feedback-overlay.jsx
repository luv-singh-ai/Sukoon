import { useState } from "react";

const env = await import.meta.env;
const API_KEY = (env.VITE_SUPABASE_API_KEY);
const AUTHORIZATION_TOKEN = (env.VITE_SUPABASE_AUTHORIZATION_TOKEN);

export default function FeedbackOverlay({feedback, setFeedback, likedMessages, setLikedMessages, dislikedMessages, setDislikedMessages}) {
  const [feedbackMessage, setFeedbackMessage] = useState("");

  const submitFeedback = async e => {
    e.preventDefault();

    try {
      const response = await fetch("https://supabase.pplus.ai/rest/v1/Feedback", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'apikey': API_KEY,
          'Authorization': AUTHORIZATION_TOKEN,
          'Prefer': 'return=minimal'
        },
        body: JSON.stringify({
          'action': feedback.feedback,
          'feedback': feedbackMessage,
        }),
      });

      if (response.status !== 201) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      setFeedback("");
      setFeedbackMessage("");

      if (feedback.feedback === "like") {
        // remove the message from the list of dislikedMessages, if it was added earlier (only for local state)
        if (dislikedMessages.includes(feedback.messageId) === true) {
          setDislikedMessages(dislikedMessages => dislikedMessages.filter(item => item !== feedback.messageId));
        }
        setLikedMessages(likedMessages => [...likedMessages, feedback.messageId]);
      } else {
        // remove the message from the list of setLikedMessages, if it was added earlier (only for local state)
        if (likedMessages.includes(feedback.messageId) === true) {
          setLikedMessages(likedMessages => likedMessages.filter(item => item !== feedback.messageId));
        }
        setDislikedMessages(dislikedMessages => [...dislikedMessages, feedback.messageId]);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  }

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50" id="feedbackModal">
      <form className="bg-white w-[90vw] max-w-3xl rounded-lg shadow-lg p-6" onSubmit={e => submitFeedback(e)}>
        <p className="text-xl font-semibold mb-4">Provide Feedback</p>
        <textarea value={feedbackMessage} onChange={e => setFeedbackMessage(e.target.value)}
          className="w-full h-24 p-3 border rounded-md resize-none"
          placeholder="Write your feedback here" />
        <div className="flex justify-end mt-4 space-x-3">
          <button onClick={() => setFeedback({})} className="px-4 py-2 bg-gray-100 rounded-md">
            Cancel
          </button>
          <button type="submit" style={{backgroundColor: "rgba(255, 229, 180, 0.7)"}}
            className="px-4 py-2 rounded-md">
            Submit
          </button>
        </div>
      </form>
    </div>
  );
}
