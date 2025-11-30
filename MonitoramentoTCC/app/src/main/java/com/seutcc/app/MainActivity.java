package com.seutcc.app;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.seutcc.app.adapters.LinhaAdapter;
import com.seutcc.app.models.Linha;
import com.seutcc.app.network.RetrofitClient;
import java.util.List;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        RecyclerView rv = findViewById(R.id.recyclerLinhas);
        rv.setLayoutManager(new LinearLayoutManager(this));

        // Busca dados do Backend
        RetrofitClient.getIoTService().getLinhas().enqueue(new Callback<List<Linha>>() {
            @Override
            public void onResponse(Call<List<Linha>> call, Response<List<Linha>> response) {
                if (response.isSuccessful() && response.body() != null) {
                    LinhaAdapter adapter = new LinhaAdapter(response.body(), linha -> {
                        // Ao clicar, vai para a tela de estações
                        Intent i = new Intent(MainActivity.this, EstacoesActivity.class);
                        i.putExtra("LINHA_ID", linha.getId());
                        i.putExtra("LINHA_NOME", linha.getNome());
                        startActivity(i);
                    });
                    rv.setAdapter(adapter);
                }
            }
            @Override
            public void onFailure(Call<List<Linha>> call, Throwable t) {
                Toast.makeText(MainActivity.this, "Erro: " + t.getMessage(), Toast.LENGTH_LONG).show();
            }
        });
    }
}