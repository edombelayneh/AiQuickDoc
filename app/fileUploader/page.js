'use client'
import React, { useState, useEffect, useRef } from "react";
import { API_URL } from '../api/backend/route';
import {
  Button,
  Card,
  CardContent,
  CardHeader,
  Typography,
  TextField,
  CircularProgress,
  Alert,
  Box,
  Grid,
  List,
  ListItem,
  ListItemText,
  Paper,
  Container,
  Avatar,
  Tabs,
  Tab,
  useTheme,
  useMediaQuery
} from "@mui/material";
import SummarizeIcon from "@mui/icons-material/Summarize";
import NoteAddIcon from "@mui/icons-material/NoteAdd";
import StyleIcon from '@mui/icons-material/Style';
import SendIcon from "@mui/icons-material/Send";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import CardActionArea from "@mui/material/CardActionArea";
import { Viewer, Worker } from '@react-pdf-viewer/core';
import '@react-pdf-viewer/core/lib/styles/index.css';
import VisibilityIcon from '@mui/icons-material/Visibility';
import FolderIcon from '@mui/icons-material/Folder';
import ChatIcon from '@mui/icons-material/Chat';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import DoneIcon from '@mui/icons-material/Done';

export default function FileUploadSummarize() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [summary, setSummary] = useState("");
  const [flashcards, setFlashcards] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [isGeneratingFlashcards, setIsGeneratingFlashcards] = useState(false);
  const [uploadSessionId, setUploadSessionId] = useState("");
  const [flipped, setFlipped] = useState([]);
  const [message, setMessage] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const [hasGreeted, setHasGreeted] = useState(false);
  const [filePreviewUrl, setFilePreviewUrl] = useState(null);
  const [isTextPreviewVisible, setIsTextPreviewVisible] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [fileUploaded, setFileUploaded] = useState(false);
  const [audioData, setAudioData] = useState("");
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioPlayer, setAudioPlayer] = useState(null);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    console.log("Flashcards state updated:", flashcards);
  }, [flashcards]);

  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    setFile(uploadedFile);
    setText('');
    setIsTextPreviewVisible(false);

    if (uploadedFile) {
      const previewUrl = URL.createObjectURL(uploadedFile);
      setFilePreviewUrl(previewUrl);
    }
  };

  const handleTextChange = (event) => {
    setText(event.target.value);
    setFile(null);
    setFilePreviewUrl(null);
    setIsTextPreviewVisible(false);
  };

  const uploadFileOrText = async () => {
    if (!file && !text) {
      setUploadStatus("Please enter text or select a file first.");
      return;
    }

    setIsUploading(true);
    const formData = new FormData();

    if (file) {
      formData.append("file", file);
    } else {
      formData.append("text", text);
    }

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setUploadStatus(
          "Uploaded successfully! You can now chat with your input."
        );
        setUploadSessionId(data.upload_session_id);
        setFileUploaded(true);
        // Send greeting message
        if (!hasGreeted) {
          sendGreeting(data.upload_session_id);
        }
      } else {
        setUploadStatus("Upload failed: " + data.message);
      }
    } catch (error) {
      setUploadStatus("An error occurred during upload.");
      console.error("Upload error:", error);
    } finally {
      setIsUploading(false);
    }
  };

  const sendGreeting = async (sessionId) => {
    try {
      const response = await fetch(`${API_URL}/chat_agent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: "Greet user EXACTLY like this: 'Hello, I'm your AiQuickDoc Assistant! I'm eager to help you understand the content of your file. Let's dive in! Is there anything specific you'd like to know?' DO NOT add anything beyond this greeting!",
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let aiResponse = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        aiResponse += chunk;

        setChatMessages((prevMessages) => [
          ...prevMessages,
          { type: "ai", content: aiResponse },
        ]);
      }

      setHasGreeted(true);
    } catch (error) {
      console.error("Greeting error:", error);
      setChatMessages((prevMessages) => [
        ...prevMessages,
        {
          type: "error",
          content: "An error occurred while sending the greeting.",
        },
      ]);
    }
  };

  const summarizeFileOrText = async () => {
    if (!file && !text) {
      setSummary("Please enter text or select a file first.");
      return;
    }

    setIsSummarizing(true);
    const formData = new FormData();

    if (file) {
      formData.append("file", file);
    } else {
      formData.append("text", text);
    }

    try {
      const response = await fetch(`${API_URL}/summarize`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setSummary(data.summaries);
        setAudioData(data.audio);
      } else {
        setSummary("Summarization failed: " + data.message);
      }
    } catch (error) {
      setSummary("An error occurred during summarization.");
      console.error("Summarization error:", error);
    } finally {
      setIsSummarizing(false);
    }
  };

  const playAudio = () => {
    if (!audioData || isPlayingAudio) return;

    const audio = new Audio(`data:audio/mp3;base64,${audioData}`);
    audio.onended = () => {
      setIsPlayingAudio(false);
      setAudioPlayer(null);
    };
    audio
      .play()
      .then(() => {
        setIsPlayingAudio(true);
        setAudioPlayer(audio);
      })
      .catch((error) => {
        console.error("Error playing audio:", error);
        setIsPlayingAudio(false);
      });
  };

  const stopAudio = () => {
    if (audioPlayer) {
      audioPlayer.pause();
      audioPlayer.currentTime = 0;
      setIsPlayingAudio(false);
      setAudioPlayer(null);
    }
  };

  useEffect(() => {
    return () => {
      if (audioPlayer) {
        audioPlayer.pause();
        audioPlayer.currentTime = 0;
      }
    };
  }, [audioPlayer]);

  const handleTextPreview = () => {
    setIsTextPreviewVisible(true);
  };

  const generateFlashcards = async () => {
    if (!file && !text) {
      setUploadStatus("Please enter text or select a file first.");
      return;
    }

    setIsGeneratingFlashcards(true);
    const formData = new FormData();

    if (file) {
      formData.append("file", file);
    } else {
      formData.append("text", text);
    }

    try {
      console.log("Sending request to generate flashcards...");
      const response = await fetch(`${API_URL}/flashcard`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.flashcards && Array.isArray(data.flashcards)) {
        console.log("Setting flashcards state:", data.flashcards);
        setFlashcards(data.flashcards);
        setUploadStatus("Flashcards generated successfully!");
      } else {
        console.error("Invalid flashcards data:", data);
        setUploadStatus("Flashcard generation failed: Invalid data format");
      }
    } catch (error) {
      console.error("Flashcard generation error:", error);
      setUploadStatus("An error occurred during flashcard generation.");
    } finally {
      setIsGeneratingFlashcards(false);
    }
  };

  const handleCardClick = (index) => {
    setFlipped((prev) =>
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index]
    );
  };

  const handleChat = async () => {
    if (!uploadSessionId || !message.trim()) {
      console.error("No session ID available or empty message");
      return;
    }

    setIsSendingMessage(true);

    try {
      const response = await fetch(`${API_URL}/chat_agent`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          session_id: uploadSessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      // Add user message immediately
      setChatMessages((prevMessages) => [
        ...prevMessages,
        { type: "user", content: message },
      ]);

      let aiResponse = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        aiResponse += chunk;

        // Update AI response in real-time
        setChatMessages((prevMessages) => {
          const lastMessage = prevMessages[prevMessages.length - 1];

          // If the last message is from AI, update its content with the new chunk
          if (lastMessage?.type === "ai") {
            lastMessage.content += chunk; // Append the new chunk to the existing content
            return [...prevMessages];
          } else {
            // If no previous AI message, add a new one
            return [...prevMessages, { type: "ai", content: chunk }];
          }
        });
      }

      // Clear the message input
      setMessage("");
    } catch (error) {
      console.error("Chat error:", error);
      // Add an error message to chat history
      setChatMessages((prevMessages) => [
        ...prevMessages,
        {
          type: "error",
          content: "An error occurred while sending the message.",
        },
      ]);
    } finally {
      setIsSendingMessage(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      backgroundColor: 'transparent'}}>
        
      <Box sx={{ 
        display: 'flex', 
        flexDirection: isMobile ? 'column' : 'row',
        flexGrow: 1,
        minHeight: '100vh',
        width: '100%',
        mt: 12,
        px: 3,
        mb: 3 }}>
        {/* Left side: Preview and Buttons */}
        <Box sx={{ width: isMobile ? '100%' : '40%', p: 2, borderRight: isMobile ? 'none' : `1px solid ${theme.palette.divider}` }}>
          {/* File Preview */}
          {(filePreviewUrl || isTextPreviewVisible) && (
              <Paper elevation={3} sx={{ mb: 3, p: 1, maxHeight: '800px', overflow: 'auto' }}>
                <Typography variant="h6" sx={{ mb: 1 }}>File Preview:</Typography>
                {filePreviewUrl && file && file.type === 'application/pdf' && (
                  <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
                    <div style={{ height: '350px' }}>
                      <Viewer fileUrl={filePreviewUrl} />
                    </div>
                  </Worker>
                )}
                {filePreviewUrl && file && file.type.startsWith('image/') && (
                  <img src={filePreviewUrl} alt="Preview" style={{ maxWidth: '100%', height: 'auto' }} />
                )}
                {filePreviewUrl && file && file.type === 'text/plain' && (
                  <iframe
                    src={filePreviewUrl}
                    style={{ width: '100%', height: '350px', border: 'none' }}
                    title="Text File Preview"
                  />
                )}
                {isTextPreviewVisible && text && (
                  <Typography sx={{ whiteSpace: 'pre-wrap' }}>
                    {text}
                  </Typography>
                )}
              </Paper>
            )}
          <Card sx={{ 
            bgcolor: 'rgba(255, 255, 255, 0)', 
            boxShadow: theme.shadows[3],
            borderRadius: '16px',
            backdropFilter: 'blur(10px)',
          }}>
            <CardHeader 
              title="File Upload or Text Input" 
              sx={{ 
                color: theme.components.contained,
                bgcolor: 'rgba(63, 81, 181, 0.1)',
                borderRadius: '16px 16px 0 0',
              }} 
            />
            <CardContent>
            <Grid container spacing={2}>
                
                <Grid item xs={12}>
                  <TextField
                    label="Enter text or choose a file"
                    multiline
                    rows={4}
                    fullWidth
                    variant="outlined"
                    value={text}
                    onChange={handleTextChange}
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    type="file"
                    fullWidth
                    onChange={handleFileChange}
                    inputProps={{ accept: ".pdf" }}
                    variant="outlined"
                  />
                </Grid>
                  <Grid item xs={12}>
                    <Button
                      variant="contained"
                      color="secondary"
                      startIcon={isSummarizing ? <CircularProgress size={24} /> : <SummarizeIcon />}
                      onClick={summarizeFileOrText}
                      disabled={isSummarizing || (!file && !text)}
                      fullWidth
                      sx={{ 
                        borderRadius: '30px',
                        textTransform: 'none',
                        fontWeight: 600,
                        boxShadow: theme.shadows[2]
                      }}
                    >
                      Summarize
                    </Button>
                  </Grid>
                  
                  <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={isUploading ? <CircularProgress size={24} /> : <ChatIcon />}
                    onClick={uploadFileOrText}
                    disabled={isUploading || (!file && !text)}
                    fullWidth
                    sx={{ 
                      borderRadius: '30px',
                      textTransform: 'none',
                      fontWeight: 600,
                      boxShadow: theme.shadows[2]
                    }}
                  >
                    Chat with File
                  </Button>
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    color="success"
                    startIcon={isGeneratingFlashcards ? <CircularProgress size={24} /> : <StyleIcon />}
                    onClick={generateFlashcards}
                    disabled={isGeneratingFlashcards || (!file && !text)}
                    fullWidth
                    sx={{ 
                      borderRadius: '30px',
                      textTransform: 'none',
                      fontWeight: 600,
                      boxShadow: theme.shadows[2]
                    }}
                  >
                    Generate Flashcards
                  </Button>
                </Grid>
                <Grid item xs={12}>
                  <Button
                    variant="outlined"
                    color="primary"
                    startIcon={<VisibilityIcon />}
                    onClick={handleTextPreview}
                    disabled={!text}
                    fullWidth
                    sx={{ 
                      borderRadius: '30px',
                      textTransform: 'none',
                      fontWeight: 600,
                      boxShadow: theme.shadows[2]
                    }}
                  >
                    Preview Text
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {uploadStatus && (
            <Alert severity="info" sx={{ mt: 2, borderRadius: '8px' }}>
              <Typography variant="body1">{uploadStatus}</Typography>
            </Alert>
          )}
        </Box>

        {/* Right side: Tabbed interface for Summary, Flashcards, and Chat */}
        <Box sx={{ width: isMobile ? '100%' : '60%', p: 2 }}>
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            aria-label="content tabs"
            sx={{
              mb: 2,
              '& .MuiTab-root': {
                borderRadius: '12px 12px 0 0',
                border: `1px solid ${theme.palette.divider}`,
                borderBottom: 'none',
                mr: 1,
                transition: 'all 0.3s',
                '&.Mui-selected': {
                  backgroundColor: theme.palette.primary.main,
                  color: theme.palette.primary.contrastText,
                  transform: 'translateY(-4px)',
                },
              },
            }}
          >
            <Tab icon={<FolderIcon />} label="Summary" />
            <Tab icon={<StyleIcon />} label="Flashcards" />
            <Tab icon={<ChatIcon />} label="Chat" />
          </Tabs>

          <Box sx={{ 
            p: 3, 
            border: `1px solid ${theme.palette.divider}`, 
            borderRadius: '0 16px 16px 16px', 
            bgcolor: 'rgba(255, 255, 255, 0.2)',
            boxShadow: theme.shadows[4],
            backdropFilter: 'blur(10px)',
          }}>
            {activeTab === 0 && (
              <Box>
                <Typography variant="h5" gutterBottom sx={{ color: theme.components.contained, fontWeight: 'bold' }}>Summary</Typography>
                {summary ? (
                  <Paper elevation={2} sx={{ p: 2, borderRadius: '12px', bgcolor: 'rgba(255, 255, 255, 0)' }}>
                    <Typography variant="body1">{summary}</Typography>
                    {/* AUDIO GOES HERE */}
                    {audioData && (
                      <Box sx={{ position: 'absolute', bottom: 10, right: 10 }}>
                        <Button
                          onClick={isPlayingAudio ? stopAudio : playAudio}
                          variant="contained"
                          color={isPlayingAudio ? "secondary" : "primary"}
                          size="small"
                          sx={{
                            minWidth: 'auto',
                            borderRadius: '20%',
                            width: 40,
                            height: 40,
                            p: 0,
                          }}
                        >
                          {isPlayingAudio ?  <StopIcon />  : <PlayArrowIcon />}
                        </Button>
                      </Box>
                    )}
                  </Paper>
                ) : (
                  <Typography variant="body2" color="text.secondary">No summary available yet. Use the Summarize button to generate a summary.</Typography>
                )}
              </Box>
            )}

            {activeTab === 1 && (
              <Box>
                <Typography variant="h5" gutterBottom sx={{ color: theme.components.contained, fontWeight: 'bold' }}>Flashcards</Typography>
                {flashcards.length > 0 ? (
                  <Grid container spacing={3}>
                    {flashcards.map((flashcard, index) => (
                      <Grid item xs={12} sm={6} md={4} key={index}>
                        <Card sx={{ 
                          boxShadow: "0 8px 24px rgba(0, 0, 0, 0.15)", 
                          bgcolor: "#ffffff", 
                          color: "#333", 
                          borderRadius: "16px", 
                          overflow: "hidden", 
                          position: "relative",
                          transition: "all 0.3s",
                          '&:hover': {
                            transform: 'translateY(-5px)',
                            boxShadow: "0 12px 32px rgba(0, 0, 0, 0.2)",
                          }
                        }}>
                          <CardActionArea onClick={() => handleCardClick(index)}>
                            <CardContent>
                              <Box sx={{
                                perspective: "1000px",
                                position: "relative",
                                width: "100%",
                                height: "200px", 
                                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
                                borderRadius: "12px",
                                overflow: "hidden",
                              }}>
                                <Box sx={{
                                  transformStyle: "preserve-3d",
                                  transition: "transform 0.6s",
                                  transform: flipped.includes(index) ? "rotateY(180deg)" : "rotateY(0deg)",
                                  position: "absolute",
                                  width: "100%",
                                  height: "100%",
                                }}>
                                  {/* Front of the Card (Question Side) */}
                                  <Box sx={{
                                    position: "absolute",
                                    width: "100%",
                                    height: "100%",
                                    backfaceVisibility: "hidden",
                                    display: "flex",
                                    justifyContent: "center",
                                    alignItems: "center",
                                    padding: 2,
                                    boxSizing: "border-box",
                                    border: "2px solid #3f51b5",
                                    textOverflow: "ellipsis",
                                    whiteSpace: "normal",
                                    overflowY: "auto",
                                    textAlign: "center",
                                    wordWrap: "break-word",
                                    background: "linear-gradient(135deg, #e8eaf6, #c5cae9)",
                                    borderRadius: "12px"
                                  }}>
                                    <HelpOutlineIcon sx={{ 
                                      fontSize: 28, 
                                      color: "#3f51b5", 
                                      position: "absolute", 
                                      top: 8, 
                                      left: 8 
                                    }} />
                                    <Typography variant="body1" align="center" sx={{
                                      fontSize: { xs: '14px', sm: '16px', md: '18px' },
                                      fontWeight: "bold",
                                      color: "#333",
                                      maxHeight: "100%",
                                      width: "fit-content",
                                      wordBreak: "break-word",
                                      overflowWrap: "break-word",
                                      p:2
                                    }}>
                                      {flashcard.front}
                                    </Typography>
                                  </Box>

                                  {/* Back of the Card (Answer Side) */}
                                  <Box sx={{
                                    position: "absolute",
                                    width: "100%",
                                    height: "100%",
                                    backfaceVisibility: "hidden",
                                    display: "flex",
                                    justifyContent: "center",
                                    alignItems: "center",
                                    padding: 2,
                                    boxSizing: "border-box",
                                    border: "2px solid #4caf50",
                                    transform: "rotateY(180deg)",
                                    background: "linear-gradient(135deg, #e8f5e9, #c8e6c9)",
                                    whiteSpace: "normal",
                                    overflowY: "auto",
                                    textAlign: "center",
                                    wordWrap: "break-word",
                                    borderRadius: "12px"
                                  }}>
                                  <DoneIcon sx={{ 
                                    fontSize: 28, 
                                    color: "#4caf50", 
                                    position: "absolute", 
                                    top: 8, 
                                    left: 8 
                                  }} />
                                  <Typography variant="body1" align="center" sx={{
                                    lineHeight: 1.8,
                                    color: "#1b5e20",
                                    fontSize: { xs: '10px', sm: '12px', md: '14px' },
                                    fontWeight: "medium",
                                    maxHeight: "100%",
                                    width: "fit-content",
                                    wordBreak: "break-word",
                                    overflowWrap: "break-word",
                                    p:2
                                  }}>
                                    {flashcard.back}
                                  </Typography>
                                </Box>
                              </Box>
                            </Box>
                          </CardContent>
                        </CardActionArea>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              ) : (
                <Typography variant="body2" color="text.secondary">No flashcards generated yet. Use the Generate Flashcards button to create flashcards.</Typography>
              )}
            </Box>
          )}

            {activeTab === 2 && (
              <Box>
                <Typography variant="h5" gutterBottom sx={{ color: theme.components.contained, fontWeight: 'bold' }}>Chat with File</Typography>
                {fileUploaded ? (
                  <Box sx={{ display: "flex", flexDirection: "column", height: "500px" }}>
                    <Paper elevation={3} sx={{ 
                      flexGrow: 1, 
                      overflow: "auto", 
                      mb: 2, 
                      p: 2, 
                      borderRadius: '16px',
                      bgcolor: 'rgba(255, 255, 255, 0)',
                      backdropFilter: 'blur(5px)',
                    }}>
                      <List>
                        {chatMessages.map((msg, index) => (
                          <ListItem key={index} sx={{
                            justifyContent: msg.type === "user" ? "flex-end" : "flex-start",
                            alignItems: "flex-start",
                            mb: 1,
                          }}>
                            {msg.type !== "user" && (
                              <Avatar
                                alt="AI Avatar"
                                src="bot.png"
                                sx={{ mr: 2, bgcolor: theme.contained }}
                              />
                            )}
                            <Paper elevation={1} sx={{
                              p: 2,
                              backgroundColor: msg.type === "user" ? theme.palette.primary.light : theme.palette.secondary.light,
                              maxWidth: "70%",
                              borderRadius: msg.type === "user" ? '20px 20px 0 20px' : '20px 20px 20px 0',
                            }}>
                              <ListItemText
                                primary={msg.type === "user" ? "You" : "AI"}
                                secondary={msg.content}
                                primaryTypographyProps={{
                                  fontWeight: "bold",
                                  color: msg.type === "user" ? theme.palette.primary.dark : theme.palette.secondary.dark,
                                }}
                                secondaryTypographyProps={{
                                  color: msg.type === "user" ? theme.palette.primary.contrastText : theme.palette.secondary.contrastText,
                                }}
                              />
                            </Paper>
                            {msg.type === "user" && (
                              <Avatar
                                alt="User Avatar"
                                src="user.png"
                                sx={{ ml: 2}}
                              />
                            )}
                          </ListItem>
                        ))}
                      </List>
                      <div ref={messagesEndRef} />
                    </Paper>
                    <Box sx={{ display: "flex", alignItems: "center" }}>
                      <TextField
                        fullWidth
                        variant="outlined"
                        placeholder="Ask something about your uploaded data..."
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyPress={(e) => {
                          if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            handleChat();
                          }
                        }}
                        disabled={isSendingMessage}
                        sx={{ 
                          mr: 1,
                          '& .MuiOutlinedInput-root': {
                            borderRadius: '25px',
                            backgroundColor: 'rgba(255, 255, 255, 0)',
                          }
                        }}
                      />
                      <Button
                        variant="contained"
                        color="primary"
                        endIcon={isSendingMessage ? <CircularProgress size={24} /> : <SendIcon />}
                        onClick={handleChat}
                        disabled={isSendingMessage || !message.trim()}
                        sx={{
                          borderRadius: '25px',
                          padding: '10px 20px',
                          textTransform: 'none',
                          fontWeight: 'bold',
                          boxShadow: theme.shadows[3],
                          '&:hover': {
                            backgroundColor: theme.palette.primary.dark,
                          },
                        }}
                      >
                        Send
                      </Button>
                    </Box>
                   </Box>
                ) : (
                  <Box>
                    <Typography variant="body2" color="text.secondary">Chat not started yet. Use the Chat with File button to start a Chat.</Typography>
                  </Box> 
                )}
              </Box>
            )}
          </Box>
        </Box>
      </Box>

      <Box component="footer" 
        sx={{
          py: 4,
          px: 2,
          mt: 'auto',
          bgcolor: 'background.paper' }}>
        <Container maxWidth="lg">
          <Typography variant="body1" align="center">
            © 2024 AIQuickDoc. All rights reserved.
          </Typography>
          <Typography variant="body2" color="text.secondary" align="center">
           Crafted with ❤️ by Edom Belayneh, Helena Tesfaye, Manizha Khorram, Mino Ralison.
          </Typography>
        </Container>
      </Box>
    </Box>
  );
}

