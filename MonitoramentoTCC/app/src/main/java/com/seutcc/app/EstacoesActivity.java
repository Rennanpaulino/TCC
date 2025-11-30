package com.seutcc.app;

import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.seutcc.app.adapters.EstacaoAdapter;
import com.seutcc.app.models.Estacao;
import com.seutcc.app.network.RetrofitClient;
import java.util.List;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class EstacoesActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_estacoes);

        String linhaId = getIntent().getStringExtra("LINHA_ID");
        String linhaNome = getIntent().getStringExtra("LINHA_NOME");

        ((TextView)findViewById(R.id.txtTituloLinha)).setText(linhaNome);

        RecyclerView rv = findViewById(R.id.recyclerEstacoes);
        rv.setLayoutManager(new LinearLayoutManager(this));

        // Busca estações dessa linha no Backend
        RetrofitClient.getIoTService().getEstacoes(linhaId).enqueue(new Callback<List<Estacao>>() {
            @Override
            public void onResponse(Call<List<Estacao>> call, Response<List<Estacao>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    EstacaoAdapter adapter = new EstacaoAdapter(response.body(), estacao -> {
                        // Vai para a tela de previsão
                        Intent i = new Intent(EstacoesActivity.this, DetalhesEstacaoActivity.class);
                        i.putExtra("LINHA_ID", linhaId);
                        i.putExtra("ESTACAO_ID", estacao.getId());
                        i.putExtra("ESTACAO_NOME", estacao.getNome());
                        startActivity(i);
                    });
                    rv.setAdapter(adapter);
                }
            }
            @Override public void onFailure(Call<List<Estacao>> call, Throwable t) {}
        });
    }
}