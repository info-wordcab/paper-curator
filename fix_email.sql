-- Update the email address for the neural_speech_decoding topic
-- Replace 'your-email@example.com' with your actual email address

UPDATE email_queue 
SET recipient_email = 'aleks@wordcab.com',  -- Change this to your desired email
    attempts = 0,                            -- Reset attempts
    last_error = NULL                        -- Clear error
WHERE topic_id = 'neural_speech_decoding' 
  AND status = 'pending';

-- Verify the update
SELECT queue_id, topic_id, topic_name, recipient_email, attempts
FROM email_queue 
WHERE topic_id = 'neural_speech_decoding';