package com.seutcc.app.models;

import com.google.gson.annotations.SerializedName;

public class LoginResponse {

    // O @SerializedName garante que o Java entenda o snake_case do Python
    @SerializedName("access_token")
    private String accessToken;

    @SerializedName("token_type")
    private String tokenType;

    // Getters (Retrofit usa isso para ler os dados)
    public String getAccessToken() {
        return accessToken;
    }

    public String getTokenType() {
        return tokenType;
    }
}