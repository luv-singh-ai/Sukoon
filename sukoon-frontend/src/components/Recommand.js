import React from 'react'

function Recommand({recommandSend}) {
    return (
        <div className='row recommandQuery justify-content-center w-100 mt-5 p-5 gx-3'>
            <div className='col-sm-1 col-md item shadow-sm rounded border px-2 py-2 ms-2' onClick={() => recommandSend("Need to talk?")}>
                <span>
                    <i className="fa-solid fa-square-plus fs-5"></i>&nbsp;&nbsp;
                    <span style={{ fontSize: 'smaller' }}>Need to talk?</span>
                </span>
            </div>
            <div className='col-sm-1 col-md item shadow-sm rounded border px-2 py-2 ms-2' onClick={() => recommandSend("How do you relax?")}>
                <i className="fa-solid fa-square-plus fs-5"></i>&nbsp;&nbsp;
                <span style={{ fontSize: 'smaller' }}>How do you relax?</span>
            </div>
            <div className='col-sm-1 col-md item shadow-sm rounded border px-2 py-2 ms-2' onClick={() => recommandSend("Stressed about exams?")}>
                <i className="fa-solid fa-square-plus fs-5"></i>&nbsp;&nbsp;
                <span style={{ fontSize: 'smaller' }}>Stressed about exams?</span>
            </div>
        </div>
    )
}

export default Recommand;