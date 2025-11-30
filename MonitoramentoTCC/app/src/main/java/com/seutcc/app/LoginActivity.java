package com.seutcc.app;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView; // <--- Não esqueça desse import
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.seutcc.app.models.LoginResponse;
import com.seutcc.app.models.UserLogin;
import com.seutcc.app.network.ApiService;
import com.seutcc.app.network.RetrofitClient;
import com.seutcc.app.utils.SessionManager;
import org.json.JSONObject;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class LoginActivity extends AppCompatActivity {

    // 1. Declaração das variáveis (View Components)
    private EditText edtUser, edtPass;
    private Button btnLogin;
    private TextView txtCadastro;
    private SessionManager session;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);

        session = new SessionManager(this);

        // Se já tem token salvo, pula o login
        if (session.isLoggedIn()) {
            abrirMain();
        }

        // 2. Ligação com o XML (FindViewById)
        edtUser = findViewById(R.id.edtUsuario);
        edtPass = findViewById(R.id.edtSenha);
        btnLogin = findViewById(R.id.btnLogin);
        txtCadastro = findViewById(R.id.txtIrParaCadastro); // <--- Inicializamos aqui

        // 3. Configuração dos Cliques

        // Botão Entrar
        btnLogin.setOnClickListener(v -> {
            String u = edtUser.getText().toString().trim();
            String p = edtPass.getText().toString().trim();

            if (!u.isEmpty() && !p.isEmpty()) {
                fazerLogin(u, p);
            } else {
                Toast.makeText(this, "Preencha usuário e senha", Toast.LENGTH_SHORT).show();
            }
        });

        // Link "Cadastre-se"
        txtCadastro.setOnClickListener(v -> {
            Intent intent = new Intent(LoginActivity.this, RegisterActivity.class);
            startActivity(intent);
        });
    }

    private void fazerLogin(String user, String pass) {
        ApiService api = RetrofitClient.getAuthService(); // Porta 8001
        UserLogin loginData = new UserLogin(user, pass);

        api.login(loginData).enqueue(new Callback<LoginResponse>() {
            @Override
            public void onResponse(Call<LoginResponse> call, Response<LoginResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    // Sucesso: Salva o token e entra
                    session.saveToken(response.body().getAccessToken());
                    abrirMain();
                } else {
                    // Erro: Tenta ler a mensagem do Backend
                    try {
                        String erroJson = response.errorBody().string();
                        JSONObject json = new JSONObject(erroJson);
                        String msg = json.optString("detail", "Credenciais Inválidas");
                        Toast.makeText(LoginActivity.this, "Falha: " + msg, Toast.LENGTH_LONG).show();
                    } catch (Exception e) {
                        Toast.makeText(LoginActivity.this, "Erro no servidor: " + response.code(), Toast.LENGTH_SHORT).show();
                    }
                }
            }

            @Override
            public void onFailure(Call<LoginResponse> call, Throwable t) {
                Toast.makeText(LoginActivity.this, "Sem conexão com o servidor.", Toast.LENGTH_LONG).show();
            }
        });
    }

    private void abrirMain() {
        Intent intent = new Intent(LoginActivity.this, MainActivity.class);
        startActivity(intent);
        finish(); // Mata a tela de Login para não voltar nela com o botão "Voltar"
    }
}