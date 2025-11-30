package com.seutcc.app.models;

public class UserLogin {
    private String username;
    private String password;

    // Construtor para facilitar a criação na LoginActivity
    public UserLogin(String username, String password) {
        this.username = username;
        this.password = password;
    }
}