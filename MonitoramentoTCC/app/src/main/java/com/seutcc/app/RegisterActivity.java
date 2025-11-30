package com.seutcc.app;

import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.seutcc.app.models.UserLogin;
import com.seutcc.app.network.RetrofitClient;
import org.json.JSONObject;
import java.util.Map;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class RegisterActivity extends AppCompatActivity {

    private EditText edtUser, edtPass;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        edtUser = findViewById(R.id.edtRegUsuario);
        edtPass = findViewById(R.id.edtRegSenha);

        Button btnCadastrar = findViewById(R.id.btnCadastrar);
        btnCadastrar.setOnClickListener(v -> tentarCadastro());

        Button btnVoltar = findViewById(R.id.btnVoltarLogin);
        btnVoltar.setOnClickListener(v -> finish()); // Fecha tela e volta pro Login
    }

    private void tentarCadastro() {
        String user = edtUser.getText().toString().trim();
        String pass = edtPass.getText().toString().trim();

        if (user.isEmpty() || pass.isEmpty()) {
            Toast.makeText(this, "Preencha todos os campos", Toast.LENGTH_SHORT).show();
            return;
        }

        UserLogin dados = new UserLogin(user, pass);

        // Chama a API de Auth (Porta 8001)
        RetrofitClient.getAuthService().register(dados).enqueue(new Callback<Map<String, String>>() {
            @Override
            public void onResponse(Call<Map<String, String>> call, Response<Map<String, String>> response) {
                if (response.isSuccessful()) {
                    Toast.makeText(RegisterActivity.this, "Conta criada! Faça login.", Toast.LENGTH_LONG).show();
                    finish(); // Volta para a tela de login automaticamente
                } else {
                    // --- AQUI ESTÁ O TRATAMENTO DE ERRO ---
                    tratarErroBackend(response);
                }
            }

            @Override
            public void onFailure(Call<Map<String, String>> call, Throwable t) {
                Toast.makeText(RegisterActivity.this, "Erro de conexão: " + t.getMessage(), Toast.LENGTH_SHORT).show();
            }
        });
    }

    // Função auxiliar para ler a mensagem de erro do JSON
    private void tratarErroBackend(Response<?> response) {
        try {
            // O backend manda: {"detail": "Usuário já existe"}
            String erroBruto = response.errorBody().string();
            JSONObject json = new JSONObject(erroBruto);
            String mensagem = json.optString("detail", "Erro desconhecido");

            Toast.makeText(this, "Erro: " + mensagem, Toast.LENGTH_LONG).show();
        } catch (Exception e) {
            Toast.makeText(this, "Erro " + response.code(), Toast.LENGTH_SHORT).show();
        }
    }
}