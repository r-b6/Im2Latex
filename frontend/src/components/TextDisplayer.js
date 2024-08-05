
import Container from 'react-bootstrap/Container';
import Form from 'react-bootstrap/Form';
import { InlineMath } from 'react-katex';
import "./TextDisplayer.css";

function TextDisplayer({includeButton, textVal, handleTextChange}){
    const handleDownload = () => {
        // Define the string you want to download
        const textToDownload = textVal;

        // Create a Blob from the string
        const blob = new Blob([textToDownload], { type: 'text/plain' });

        // Create a URL for the Blob
        const url = window.URL.createObjectURL(blob);

        // Create an anchor element and trigger a download
        const a = document.createElement('a');
        a.download = 'output.txt'; // The name of the file to be downloaded
        a.href = url;
        document.body.appendChild(a);
        a.click();

        // Cleanup: remove the anchor element and revoke the object URL
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        console.log('done')
    };

    const handleCopy = () => {
        navigator.clipboard.writeText(textVal)}

return(
        
        <Container>
            <Form.Group className="mb-3 output-box" controlId="exampleForm.ControlTextarea1">
                <Form.Label>Output:</Form.Label>
                <Form.Control 
                className="output-box-control" 
                as="textarea" 
                rows={4} 
                value={textVal}
                onChange={handleTextChange}/>
                </Form.Group>
                {includeButton &&( 
                <button className={textVal.length > 0 ? "save-btn btn btn-primary " : "save-btn btn-secondary"}
                disabled={textVal.length === 0 ? true : false}
                onClick={textVal.length > 0 ? handleDownload: () => null}
                >
                    Download as .txt
                     </button>)}
                {includeButton &&(
                <button className={textVal.length > 0 ? "copy-btn btn btn-primary " : "copy-btn btn-secondary "}
                disabled={textVal.length === 0 ? true : false}
                onClick={textVal.length > 0 ? handleCopy: () => null}
                >
                    Copy to clipboard
                     </button>)}
                <Form.Group className="mb-3 compiled output-box" controlId="exampleForm.ControlTextarea2">
                <Form.Label>Compiled Output:</Form.Label>
                {/* <Form.Control 
                className="compiled output-box-control" 
                as="textarea" 
                rows={4}
                readOnly
                /> */}
                <div>
                {textVal.includes('error') ? textVal : <InlineMath math={textVal}></InlineMath>}
                </div>
                </Form.Group>
        </Container>
        )
    }

export default TextDisplayer