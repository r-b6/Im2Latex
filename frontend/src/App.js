import './App.css';
import NavBar from './components/NavBar';
import { BrowserRouter as Router, Routes, Route} from 'react-router-dom';
import Home from './components/pages/Home';
import Create from "./components/pages/Create";
import Tutorial from "./components/pages/Tutorial";
import 'katex/dist/katex.min.css';

function App() {
  return (
    <>
      <Router>
        <NavBar />
          <Routes>
            <Route path='/' exact element={<Home/>}/>
            <Route path='/create' element={<Create/>}/>
            <Route path='/tutorial' element={<Tutorial></Tutorial>}></Route>
          </Routes>
      </Router>
    </>
  );
}

export default App;
