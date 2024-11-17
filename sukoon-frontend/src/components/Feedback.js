import React, { useState } from 'react';
import Form from 'react-bootstrap/Form';
import { Modal, Button } from 'react-bootstrap';



const Feedback = ({ show, setShowModal, act, thumbUpBtn, thumbDownBtn, status, setStatus }) => {
  const [feedback, setFeedback] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const handleClose = () => {
    if (status.status !== "success") {
      thumbDownBtn.current.classList.remove('text-warning')
      thumbUpBtn.current.classList.remove('text-warning')
    }
    setShowModal(false)
  };

  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent page reload on form submit
    setIsSubmitting(true);
    setStatus({ status: "wait", message: "" });

    // Data to send to the API
    if (!feedback) {
      setStatus({ status: "wait", message: "Please write what you think." });
      setIsSubmitting(false);
      return;
    }
    const data = {
      feedback: feedback,
      action: act
    };

    // API URL and headers
    const apiUrl = 'https://supabase.pplus.ai/rest/v1/Feedback';
    const headers = {
      'Content-Type': 'application/json',
      'apikey': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ewogICJyb2xlIjogImFub24iLAogICJpc3MiOiAic3VwYWJhc2UiLAogICJpYXQiOiAxNzIyNzk2MjAwLAogICJleHAiOiAxODgwNTYyNjAwCn0.uDIm9z3_KdGKpnEhkOA4P7RDT_xmzjO3pZnPvpzEBs8', // Replace with your actual API key
      'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ewogICJyb2xlIjogImFub24iLAogICJpc3MiOiAic3VwYWJhc2UiLAogICJpYXQiOiAxNzIyNzk2MjAwLAogICJleHAiOiAxODgwNTYyNjAwCn0.uDIm9z3_KdGKpnEhkOA4P7RDT_xmzjO3pZnPvpzEBs8', // Replace with your actual auth token
    };

    try {
      // Sending POST request to Supabase API
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(data),
      });

      // Handle response
      if (response.ok) {
        setStatus({ status: "success", message: "Feedback submitted successfully!" });
      } else {
        setStatus({ status: "error", message: "Error submitting feedback!" });
      }
    } catch (error) {
      console.error('Error:', error);
      setStatus({ status: "error", message: "Error submitting feedback!" });
    } finally {
      setIsSubmitting(false);
    }
  };
  return (
    <Modal show={show} onHide={handleClose} size="md"
      aria-labelledby="contained-modal-title-vcenter"
      centered>
      <Modal.Body className={status.status === "success" ? "text-center p-4 pt-5" : ""}>
        {status.status !== "success" ?
          <>
            <Form.Group
              className="mb-3"
              controlId="exampleForm.ControlTextarea1"
            >
              <h5 className='mb-4'>Did you find this helpful?</h5>
              <Form.Control as="textarea" rows={3} placeholder='Tell us what you think.' value={feedback} onChange={(e) => setFeedback(e.target.value)} />
              <span className='text-danger'>{status.message}</span>
            </Form.Group>
            <Button variant="primary" onClick={handleSubmit} className='float-end' disabled={isSubmitting}>
              Send Feedback
            </Button>
          </>
          :
          <>
            <Form.Group
              className="mb-3"
              controlId="exampleForm.ControlTextarea1"
            >
              <h5 className='mb-4'>Thank you for giving your feedback.</h5>
            </Form.Group>
            <Button variant="primary" onClick={handleClose} className='mt-3'>
              Done
            </Button>
          </>
        }
      </Modal.Body>
    </Modal>
  );
};

export default Feedback;
