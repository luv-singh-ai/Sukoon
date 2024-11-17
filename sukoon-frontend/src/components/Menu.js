import React from 'react'

function Menu() {
    return (
        <>
            <div className="menu">
                <div className="button" style={{ textAlign: "center" }}>
                    <i className="fa-regular fa-square-plus" style={{ fontSize: "20px" }}></i>
                    <p style={{ margin: 0 }}>New</p>
                </div>
                <div className="button" style={{ textAlign: "center" }}>
                    <i className="fa-solid fa-wand-magic-sparkles" style={{ fontSize: "20px" }}></i>
                    <p style={{ margin: 0 }}>Discover</p>
                </div>
                <div className="button" style={{ textAlign: "center" }}>
                    <i className="fa-regular fa-user" style={{ fontSize: "20px" }}></i>
                    <p style={{ margin: 0 }}>Profile</p>
                </div>
            </div>
            <div className="menu-content">
                <h4 style={{ padding: "5px" }}>Discover Chats</h4>
                <div className="row discover">
                    <div className="col-6">
                        {/* <img src="./img/image3.png" alt="Discover 2" /> */}
                        <img src="./img/img1.png" alt="" />
                    </div>
                    <div className="col-6">
                        <img src="./img/image3.png" alt="" />
                    </div>
                    <div className="col-12">
                        <img src="./img/Frame 4.png" alt="" />
                    </div>
                </div>
            </div>
        </>
    )
}

export default Menu;