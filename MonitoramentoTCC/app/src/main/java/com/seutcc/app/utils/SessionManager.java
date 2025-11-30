package com.seutcc.app.utils;

import android.content.Context;
import android.content.SharedPreferences;

public class SessionManager {
    private SharedPreferences prefs;
    private SharedPreferences.Editor editor;

    public SessionManager(Context context) {
        prefs = context.getSharedPreferences("TCC_PREFS", Context.MODE_PRIVATE);
        editor = prefs.edit();
    }

    public void saveToken(String token) {
        editor.putString("JWT_TOKEN", token);
        editor.apply();
    }

    public String getToken() {
        return prefs.getString("JWT_TOKEN", null);
    }

    public boolean isLoggedIn() {
        return getToken() != null;
    }

    public void logout() {
        editor.clear();
        editor.apply();
    }
}